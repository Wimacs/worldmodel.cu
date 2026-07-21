#include "world_control.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(condition, message) do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__); \
            failures++; \
        } \
    } while (0)

static int close_float(float actual, float expected) {
    return fabsf(actual - expected) <= 1.0e-5f;
}

static void test_mouse_velocity_range_and_ema(void) {
    WorldMouseVelocityState state;
    world_mouse_velocity_reset(&state, 0.0);

    world_mouse_velocity_update(&state, 10.0f * WORLD_MOUSE_DEFAULT_SCALE, 0.0f, 1.0f / 60.0f, 0.0);
    CHECK(close_float(state.x, 5.25f), "first EMA update should match official gain and alpha");
    CHECK(state.x > 1.0f, "mouse velocity must not be clamped to one");

    world_mouse_velocity_update(&state, 10.0f * WORLD_MOUSE_DEFAULT_SCALE, 0.0f, 1.0f / 60.0f, 0.01);
    CHECK(close_float(state.x, 8.6625f), "mouse state should track latest velocity instead of accumulating displacement");
}

static void test_mouse_idle_decay(void) {
    WorldMouseVelocityState state;
    world_mouse_velocity_reset(&state, 0.0);
    world_mouse_velocity_update(&state, 10.0f, -4.0f, 1.0f / 60.0f, 0.0);
    float active_x = state.x;
    float active_y = state.y;

    world_mouse_velocity_update(&state, 0.0f, 0.0f, 1.0f / 30.0f, 0.04);
    CHECK(close_float(state.x, active_x) && close_float(state.y, active_y),
          "mouse velocity should be held for the first 50 ms");

    world_mouse_velocity_update(&state, 0.0f, 0.0f, 1.0f / 30.0f, 0.06);
    CHECK(close_float(state.x, active_x * WORLD_MOUSE_DECAY_FACTOR),
          "idle mouse x should decay at the official 30 Hz factor");
    CHECK(close_float(state.y, active_y * WORLD_MOUSE_DECAY_FACTOR),
          "idle mouse y should decay at the official 30 Hz factor");

    state.x = 0.009f;
    state.y = -0.009f;
    state.decay_seconds = 0.0f;
    state.last_motion_time = 0.0;
    world_mouse_velocity_update(&state, 0.0f, 0.0f, 1.0f / 30.0f, 0.10);
    CHECK(state.x == 0.0f && state.y == 0.0f, "tiny idle velocity should settle exactly to zero");
}

static void test_mouse_reset(void) {
    WorldMouseVelocityState state = {3.0f, -2.0f, 0.25f, 1.0};
    world_mouse_velocity_reset(&state, 5.0);
    CHECK(state.x == 0.0f && state.y == 0.0f, "reset should clear both mouse axes");
    CHECK(state.decay_seconds == 0.0f && state.last_motion_time == 5.0,
          "reset should restart idle timing");
}

int main(void) {
    test_mouse_velocity_range_and_ema();
    test_mouse_idle_decay();
    test_mouse_reset();
    if (failures) {
        fprintf(stderr, "%d world control test(s) failed\n", failures);
        return 1;
    }
    fprintf(stderr, "world control tests passed\n");
    return 0;
}
