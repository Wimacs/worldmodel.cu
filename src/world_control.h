#ifndef WORLD_CONTROL_H
#define WORLD_CONTROL_H

#include <math.h>

#define WORLD_MOUSE_DEFAULT_SCALE 1.5f
#define WORLD_MOUSE_EMA_ALPHA 0.35f
#define WORLD_MOUSE_IDLE_SECONDS 0.05
#define WORLD_MOUSE_DECAY_INTERVAL_SECONDS (1.0f / 30.0f)
#define WORLD_MOUSE_DECAY_FACTOR 0.7f
#define WORLD_MOUSE_ZERO_THRESHOLD 0.01f

typedef struct {
    float x;
    float y;
    float decay_seconds;
    double last_motion_time;
} WorldMouseVelocityState;

static inline void world_mouse_velocity_reset(WorldMouseVelocityState *state, double now) {
    state->x = 0.0f;
    state->y = 0.0f;
    state->decay_seconds = 0.0f;
    state->last_motion_time = now;
}

static inline void world_mouse_velocity_update(
        WorldMouseVelocityState *state,
        float target_x,
        float target_y,
        float frame_seconds,
        double now) {
    if (target_x != 0.0f || target_y != 0.0f) {
        state->x += (target_x - state->x) * WORLD_MOUSE_EMA_ALPHA;
        state->y += (target_y - state->y) * WORLD_MOUSE_EMA_ALPHA;
        state->decay_seconds = 0.0f;
        state->last_motion_time = now;
        return;
    }

    if (now - state->last_motion_time <= WORLD_MOUSE_IDLE_SECONDS) return;
    if (frame_seconds > 0.0f) state->decay_seconds += fminf(frame_seconds, 0.25f);
    int decay_steps = (int)(state->decay_seconds / WORLD_MOUSE_DECAY_INTERVAL_SECONDS);
    if (decay_steps <= 0) return;
    state->decay_seconds -= (float)decay_steps * WORLD_MOUSE_DECAY_INTERVAL_SECONDS;
    float decay = powf(WORLD_MOUSE_DECAY_FACTOR, (float)decay_steps);
    state->x *= decay;
    state->y *= decay;
    if (fabsf(state->x) < WORLD_MOUSE_ZERO_THRESHOLD) state->x = 0.0f;
    if (fabsf(state->y) < WORLD_MOUSE_ZERO_THRESHOLD) state->y = 0.0f;
}

#endif
