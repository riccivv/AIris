"""AIris benchmark suite.

Benchmarks the core components of the AIris assistant without requiring a
camera or microphone. All heavy components are exercised offline using
synthetic frames and a stubbed audio manager; the OpenAI client is replaced
with a fake so no network calls or API spend happen unless --live-api is set.

Usage:
    python benchmarks/run_benchmark.py                 # full offline suite
    python benchmarks/run_benchmark.py --quick         # reduced iterations
    python benchmarks/run_benchmark.py --skip-yolo     # skip YOLO inference
    python benchmarks/run_benchmark.py --live-api      # also hit OpenAI (costs money)
    python benchmarks/run_benchmark.py --json out.json # machine-readable report
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from airis.config import TEXT_MODEL, VISION_MODEL, WALKING_SPEED_MPS, YOLO_MODEL_PATH
from airis.obstacle_avoidance import ObstacleAvoidance
from airis.navigation import NavigationManager
from airis.llm_manager import LLMManager
from airis.conversation_manager import ConversationManager
from airis.vision_ai import VisionAI


class StubAudio:
    def __init__(self):
        self.spoken = []

    def speak(self, text, force=False):
        self.spoken.append(text)


def percentile(samples, p):
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def time_it(fn, iterations, *args, **kwargs):
    samples = []
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def summarize(name, samples):
    mean = statistics.fmean(samples)
    return {
        "name": name,
        "samples": len(samples),
        "min_ms": round(min(samples), 3),
        "p50_ms": round(percentile(samples, 50), 3),
        "mean_ms": round(mean, 3),
        "p95_ms": round(percentile(samples, 95), 3),
        "max_ms": round(max(samples), 3),
        "ops_per_sec": round(1000.0 / mean, 1) if mean > 0 else float("inf"),
    }


def make_blank_frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_obstructed_left_frame(w=640, h=480, seed=7):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    frame[:, :w // 3] = rng.integers(0, 256, size=(h, w // 3, 3), dtype=np.uint8)
    return frame


def make_noise_frame(w=640, h=480, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def make_scene_frame(w=640, h=480):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 80), (220, 400), (180, 160, 140), -1)
    cv2.rectangle(frame, (260, 120), (420, 380), (90, 110, 130), -1)
    cv2.circle(frame, (520, 240), 90, (200, 200, 200), -1)
    cv2.putText(frame, "EXIT", (280, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return frame


class SuiteResult:
    def __init__(self, name):
        self.name = name
        self.metrics = []
        self.checks = []

    def metric(self, name, samples):
        entry = summarize(name, samples)
        self.metrics.append(entry)
        return entry

    def check(self, name, passed, detail=""):
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        return passed

    @property
    def passed(self):
        return all(c["passed"] for c in self.checks)


def print_suite(result):
    print(f"\n[{result.name}]")
    header = f"  {'benchmark':<38}{'n':>6}{'min':>10}{'p50':>10}{'mean':>10}{'p95':>10}{'max':>10}{'ops/s':>10}"
    if result.metrics:
        print(header)
        print("  " + "-" * (len(header) - 2))
        for m in result.metrics:
            print(
                f"  {m['name']:<38}{m['samples']:>6}{m['min_ms']:>10.3f}{m['p50_ms']:>10.3f}"
                f"{m['mean_ms']:>10.3f}{m['p95_ms']:>10.3f}{m['max_ms']:>10.3f}{m['ops_per_sec']:>10.1f}"
            )
    for c in result.checks:
        status = "PASS" if c["passed"] else "FAIL"
        detail = f" ({c['detail']})" if c["detail"] else ""
        print(f"  [{status}] {c['name']}{detail}")
    passed = sum(1 for c in result.checks if c["passed"])
    print(f"  checks: {passed}/{len(result.checks)}")


def run_obstacle_suite(iterations):
    suite = SuiteResult("obstacle_avoidance")
    oa = ObstacleAvoidance()

    blank = make_blank_frame()
    obstructed = make_obstructed_left_frame()
    noisy = make_noise_frame()

    analysis_blank = oa.analyze_scene(blank.copy())
    suite.check("blank frame -> no avoidance", analysis_blank["should_avoid"] is False,
                f"densities={ {k: round(float(v), 1) for k, v in analysis_blank['obstacles'].items()} }")

    analysis_obstructed = oa.analyze_scene(obstructed.copy())
    suite.check("obstructed left third -> avoidance triggered", analysis_obstructed["should_avoid"] is True,
                f"left={analysis_obstructed['obstacles']['left']:.1f}")
    suite.check("obstructed left third -> clearest path avoids left",
                analysis_obstructed["clearest_path"] in ("center", "right"),
                f"clearest={analysis_obstructed['clearest_path']}")

    instruction = oa.get_avoidance_instruction(analysis_obstructed)
    suite.check("avoidance instruction issued", instruction is not None, f"instruction={instruction!r}")
    suppressed = oa.get_avoidance_instruction(analysis_obstructed)
    suite.check("cooldown suppresses repeat instruction", suppressed is None)

    suite.metric("analyze_scene blank 640x480", time_it(oa.analyze_scene, iterations, blank.copy()))
    suite.metric("analyze_scene obstructed 640x480", time_it(oa.analyze_scene, iterations, obstructed.copy()))
    suite.metric("analyze_scene noise 640x480", time_it(oa.analyze_scene, iterations, noisy.copy()))
    return suite


def build_route():
    return [
        {"instruction": "Head straight for 26 meters", "distance_meters": 26.0, "turn": "Continue straight"},
        {"instruction": "Walk 15 meters", "distance_meters": 15.0, "turn": "Turn left"},
        {"instruction": "Walk 13 meters", "distance_meters": 13.0, "turn": "Turn right onto Main Street"},
    ]


def simulate_route(route, dt=0.25, max_steps=20000):
    audio = StubAudio()
    nav = NavigationManager(audio)
    destination = {"name": "Main Street Cafe", "route": route}
    started = nav.start_route(destination)
    anchor = time.time()
    sim_elapsed = 0.0
    prev_step = nav.current_step_index
    update_samples = []
    steps = 0
    while not nav.route_completed and steps < max_steps:
        t0 = time.perf_counter()
        nav.update(anchor + sim_elapsed)
        update_samples.append((time.perf_counter() - t0) * 1000.0)
        sim_elapsed += dt
        steps += 1
        if nav.current_step_index != prev_step:
            prev_step = nav.current_step_index
            anchor = time.time()
            sim_elapsed = 0.0
    return nav, audio, update_samples, started


def run_navigation_suite():
    suite = SuiteResult("navigation")
    route = build_route()
    nav, audio, update_samples, started = simulate_route(route)

    suite.check("route starts", started is True)
    suite.check("route completes", nav.route_completed is True,
                f"steps={len(route)}, total={sum(s['distance_meters'] for s in route):.0f}m")

    spoken = " | ".join(audio.spoken)
    n_preturn = sum(1 for s in audio.spoken if "In 10 meters" in s)
    n_turn_now = sum(1 for s in audio.spoken if "now." in s)
    n_start = sum(1 for s in audio.spoken if "Starting route" in s)
    n_done = sum(1 for s in audio.spoken if "Route completed" in s)
    suite.check("start announced once", n_start == 1, f"count={n_start}")
    suite.check("pre-turn alerts exactly per turn step", n_preturn == 2, f"count={n_preturn}")
    suite.check("turn-now alerts exactly per turn step", n_turn_now == 2, f"count={n_turn_now}")
    suite.check("completion announced once", n_done == 1, f"count={n_done}")
    suite.check("total announcements exact", len(audio.spoken) == 6, f"spoken={len(audio.spoken)}")

    preturn_pos = next((i for i, s in enumerate(audio.spoken) if "In 10 meters" in s), -1)
    turn_pos = next((i for i, s in enumerate(audio.spoken) if "now." in s), -1)
    suite.check("pre-turn precedes turn-now", 0 <= preturn_pos < turn_pos,
                f"order={preturn_pos}->{turn_pos}")

    audio2 = StubAudio()
    nav2 = NavigationManager(audio2)
    short_route = [{"instruction": "Head straight for 20 meters", "distance_meters": 20.0, "turn": "You have arrived"}]
    nav2.start_route({"name": "Mailbox", "route": short_route})
    base = time.time()
    elapsed = 5.0
    nav2.update(base + elapsed)
    remaining_at_pause = nav2.remaining_distance
    nav2.pause_navigation()
    frozen = [nav2.update(base + elapsed + i * 0.5) for i in range(4)]
    suite.check("paused updates freeze progress",
                all(abs(f - remaining_at_pause) < 1e-9 for f in frozen),
                f"frozen={[round(f, 2) for f in frozen]}")
    nav2.pause_time -= 2.0
    nav2.resume_navigation()
    after = nav2.update(base + elapsed + 0.1)
    expected = max(0.0, 20.0 - ((elapsed + 0.1 - 2.0) * WALKING_SPEED_MPS))
    suite.check("resume excludes pause duration", abs(after - expected) < 0.5,
                f"got={after:.2f}m expected={expected:.2f}m")

    suite.metric("navigation update() call", update_samples)
    return suite


def make_fake_llm(temp_dir):
    llm = LLMManager(None)
    llm.cache_file = os.path.join(temp_dir, "bench_llm_cache.json")
    llm.cache = {}
    return llm


class FakeCompletions:
    def __init__(self, fail_text_model=False):
        self.calls = []
        self.fail_text_model = fail_text_model

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_text_model and kwargs.get("model") == TEXT_MODEL:
            raise RuntimeError("simulated text-model outage")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="benchmark ok"))]
        )


def install_fake_client(llm, fail_text_model=False):
    fake = FakeCompletions(fail_text_model=fail_text_model)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return fake


def run_llm_suite(iterations, temp_dir):
    suite = SuiteResult("llm_manager")
    llm = make_fake_llm(temp_dir)

    img_a = make_noise_frame(64, 64, seed=1)
    img_b = make_noise_frame(64, 64, seed=2)
    key_same_1 = llm._generate_cache_key("hello", img_a)
    key_same_2 = llm._generate_cache_key("hello", img_a)
    key_diff_prompt = llm._generate_cache_key("world", img_a)
    key_diff_image = llm._generate_cache_key("hello", img_b)
    suite.check("cache key deterministic", key_same_1 == key_same_2)
    suite.check("cache key varies by prompt", key_same_1 != key_diff_prompt)
    suite.check("cache key varies by image", key_same_1 != key_diff_image)

    optimized = llm.optimize_prompt("  hello   world  ")
    suite.check("optimize_prompt normalizes whitespace and punctuation",
                optimized == "hello world.", f"got={optimized!r}")
    long_prompt = "x" * 600
    suite.check("optimize_prompt truncates long input",
                len(llm.optimize_prompt(long_prompt)) <= 500 and llm.optimize_prompt(long_prompt).endswith("..."))

    now = time.time()
    llm.request_times = [now] * llm.rate_limit
    suite.check("rate limiter blocks at limit", llm._check_rate_limit() is False)
    llm.request_times = [now - 61.0] * llm.rate_limit
    suite.check("rate limiter resets after window", llm._check_rate_limit() is True)

    llm.request_times = []
    fake = install_fake_client(llm)
    first = llm.query_text("ping", use_cache=True)
    second = llm.query_text("ping", use_cache=True)
    suite.check("identical queries served from cache",
                first == second and len(fake.calls) == 1 and llm.cache_hits == 1,
                f"api_calls={len(fake.calls)} cache_hits={llm.cache_hits}")

    llm2 = make_fake_llm(temp_dir)
    failing_fake = install_fake_client(llm2, fail_text_model=True)
    fallback_response = llm2.query_text("ping", use_cache=False)
    suite.check("fallback model engages on primary failure",
                fallback_response == "benchmark ok" and llm2.model_usage.get(VISION_MODEL, 0) >= 1,
                f"response={fallback_response!r} usage={llm2.model_usage}")
    suite.check("failed primary attempt recorded", len(failing_fake.calls) == 2,
                f"attempts={len(failing_fake.calls)}")

    vision_stub = StubAudio()
    va = VisionAI(vision_stub, llm_manager=llm)
    stats = va.get_stats()
    suite.check("vision ai exposes llm stats", stats.get("text_model") == TEXT_MODEL,
                f"text_model={stats.get('text_model')}")

    llm.request_times = []
    llm.rate_limit = 10**9

    small_img = make_noise_frame(64, 64, seed=3)
    suite.metric("cache key gen (with 64x64 image)", time_it(llm._generate_cache_key, iterations, "prompt", small_img))
    suite.metric("optimize_prompt 600 chars", time_it(llm.optimize_prompt, iterations, long_prompt))
    suite.metric("cached query path", time_it(llm.query_text, iterations, "ping", None, True))
    suite.metric("uncached query path (fake api)", time_it(llm.query_text, iterations, "unique", None, False))
    return suite


def run_conversation_suite(iterations, temp_dir):
    suite = SuiteResult("conversation")
    llm = make_fake_llm(temp_dir)
    install_fake_client(llm)
    audio = StubAudio()
    cm = ConversationManager(audio, llm_manager=llm)
    cm.conversation_history = []

    visual_cases = [("what do you see right now", True), ("read this sign for me", True),
                    ("describe the room", True), ("what is the capital of france", False),
                    ("tell me a joke", False)]
    mismatches = [(text, cm._should_include_image(text)) for text, expected in visual_cases
                  if cm._should_include_image(text) != expected]
    suite.check("visual intent detection", not mismatches, f"mismatches={mismatches}")

    cm._update_context("My name is Jordan.", "Nice to meet you!")
    suite.check("name extraction", cm.context["user_name"] == "Jordan",
                f"name={cm.context['user_name']!r}")
    cm._update_context("I am so happy today", "Great!")
    suite.check("mood detection positive", cm.context["mood"] == "positive",
                f"mood={cm.context['mood']!r}")
    cm._update_context("how is the weather", "Sunny.")
    suite.check("topic tracking", cm.context["last_topic"] == "weather",
                f"topic={cm.context['last_topic']!r}")

    suite.check("greeting handled", cm.handle_greeting("hello there") is not None)
    suite.check("thanks handled", cm.handle_thanks("thank you so much") is not None)
    suite.check("goodbye handled", cm.handle_goodbye("good night") is not None)
    suite.check("non-greeting ignored", cm.handle_greeting("navigate to the cafe") is None)

    for i in range(50):
        cm.conversation_history.append({"user": f"question {i}", "assistant": f"answer {i}", "timestamp": time.time()})
    system_prompt = cm._build_system_prompt()
    suite.check("system prompt embeds context", "Jordan" in system_prompt and "positive" in system_prompt)
    history_prompt = cm._build_history_prompt()
    suite.check("history prompt bounded to 5 exchanges", history_prompt.count("User:") == 5,
                f"exchanges={history_prompt.count('User:')}")

    suite.metric("build system prompt", time_it(cm._build_system_prompt, iterations))
    suite.metric("build history prompt (50 entries)", time_it(cm._build_history_prompt, iterations))
    suite.metric("visual intent detection", time_it(cm._should_include_image, iterations, "what do you see here"))
    return suite


def run_yolo_suite(iterations):
    suite = SuiteResult("yolo_inference")
    from ultralytics import YOLO

    if not os.path.exists(YOLO_MODEL_PATH):
        suite.check("model weights present", False, f"missing={YOLO_MODEL_PATH}")
        return suite

    model = YOLO(YOLO_MODEL_PATH)
    frame = make_scene_frame()
    suite.check("model weights present", True, YOLO_MODEL_PATH)

    for _ in range(2):
        warm = model.predict(frame, conf=0.45, verbose=False)
    suite.check("inference returns results", len(warm) == 1, f"results={len(warm)}")

    samples = []
    detections = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        results = model.predict(frame, conf=0.45, verbose=False)
        samples.append((time.perf_counter() - t0) * 1000.0)
        detections += len(results[0].boxes)
    entry = suite.metric(f"predict 640x480 conf=0.45", samples)
    print(f"  info: avg detections/frame={detections / iterations:.1f} device=cpu")

    fps = 1000.0 / entry["mean_ms"]
    suite.check("real-time capable (>5 fps on cpu)", fps > 5.0, f"fps={fps:.1f}")
    return suite


def run_live_api_suite(temp_dir):
    suite = SuiteResult("live_api")
    llm = make_fake_llm(temp_dir)

    try:
        text_samples = []
        response = None
        for _ in range(3):
            t0 = time.perf_counter()
            response = llm.query_text("Reply with exactly: AIris benchmark OK", use_cache=False)
            text_samples.append((time.perf_counter() - t0) * 1000.0)
        suite.metric("text query (gpt-4.1-nano)", text_samples)
        suite.check("text query returns content", bool(response and response.strip()),
                    f"response={str(response)[:60]!r}")

        frame = make_scene_frame()
        t0 = time.perf_counter()
        vision_response = llm.query_vision("Describe this image in five words.", frame, use_cache=False)
        suite.metric("vision query (gpt-4o-mini)", [(time.perf_counter() - t0) * 1000.0])
        suite.check("vision query returns content", bool(vision_response and vision_response.strip()),
                    f"response={str(vision_response)[:60]!r}")
    except Exception as exc:
        suite.check("live api reachable", False, f"error={exc}")
    return suite


def main():
    parser = argparse.ArgumentParser(description="AIris benchmark suite")
    parser.add_argument("--quick", action="store_true", help="reduced iteration counts")
    parser.add_argument("--skip-yolo", action="store_true", help="skip YOLO inference benchmark")
    parser.add_argument("--live-api", action="store_true", help="include real OpenAI API calls (costs money)")
    parser.add_argument("--json", dest="json_path", help="write report to a JSON file")
    args = parser.parse_args()

    scale = 0.2 if args.quick else 1.0
    iterations = max(5, int(100 * scale))
    micro_iterations = max(500, int(5000 * scale))
    yolo_iterations = 3 if args.quick else 10

    print("=" * 60)
    print(" AIris Benchmark Suite")
    print("=" * 60)
    print(f"python={sys.version.split()[0]} opencv={cv2.__version__} quick={args.quick}")

    started_at = time.time()
    suites = []

    with tempfile.TemporaryDirectory(prefix="airis_bench_") as temp_dir:
        suites.append(run_obstacle_suite(iterations))
        suites.append(run_navigation_suite())
        suites.append(run_llm_suite(micro_iterations, temp_dir))
        suites.append(run_conversation_suite(micro_iterations, temp_dir))
        if not args.skip_yolo:
            print("\npreparing yolo suite (loading ultralytics)...")
            suites.append(run_yolo_suite(yolo_iterations))
        if args.live_api:
            suites.append(run_live_api_suite(temp_dir))

    total_checks = sum(len(s.checks) for s in suites)
    failed_checks = sum(1 for s in suites for c in s.checks if not c["passed"])

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    for s in suites:
        print_suite(s)

    elapsed = time.time() - started_at
    status = "PASS" if failed_checks == 0 else "FAIL"
    print("\n" + "=" * 60)
    print(f" overall: {status}  ({total_checks - failed_checks}/{total_checks} checks passed)")
    print(f" wall time: {elapsed:.1f}s")
    print("=" * 60)

    if args.json_path:
        report = {
            "project": "AIris",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_time_s": round(elapsed, 2),
            "mode": "quick" if args.quick else "full",
            "suites": [
                {"name": s.name, "metrics": s.metrics, "checks": s.checks, "passed": s.passed}
                for s in suites
            ],
            "summary": {
                "status": status,
                "checks_total": total_checks,
                "checks_failed": failed_checks,
            },
        }
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"report written to {args.json_path}")

    return 0 if failed_checks == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
