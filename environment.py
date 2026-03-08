import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class DoublePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, reward_type="shaped", render_mode=None):
        super().__init__()
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 600
        self.dt = 1.0 / 60.0
        self.force_magnitude = 500
        self.cart_y = 400
        self.track_limit = 300
        self.pole1_length = 100
        self.pole2_length = 100
        self.max_steps = 1000
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=np.array([-1.0, -5.0, -math.pi, -10.0, -math.pi, -10.0], dtype=np.float32),
            high=np.array([1.0, 5.0, math.pi, 10.0, math.pi, 10.0], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.screen = None
        self.clock = None
        self.space = None
        self.cart_body = None
        self.pole1_body = None
        self.pole2_body = None

        self.reset()

    def _create_space(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, -980)

        cart_mass = 1.0
        cart_moment = pymunk.moment_for_box(cart_mass, (80, 20))
        self.cart_body = pymunk.Body(cart_mass, cart_moment)
        self.cart_body.position = (self.screen_width / 2, self.screen_height - self.cart_y)
        cart_shape = pymunk.Poly.create_box(self.cart_body, (80, 20))
        cart_shape.friction = 0.0
        cart_shape.filter = pymunk.ShapeFilter(group=1)

        pole1_mass = 0.1
        pole1_moment = pymunk.moment_for_segment(pole1_mass, (0, 0), (0, self.pole1_length), 5)
        self.pole1_body = pymunk.Body(pole1_mass, pole1_moment)
        self.pole1_body.position = (self.cart_body.position.x, self.cart_body.position.y)
        pole1_shape = pymunk.Segment(self.pole1_body, (0, 0), (0, self.pole1_length), 5)
        pole1_shape.friction = 0.0
        pole1_shape.filter = pymunk.ShapeFilter(group=1)

        pole2_mass = 0.1
        pole2_moment = pymunk.moment_for_segment(pole2_mass, (0, 0), (0, self.pole2_length), 5)
        self.pole2_body = pymunk.Body(pole2_mass, pole2_moment)
        self.pole2_body.position = (self.cart_body.position.x, self.cart_body.position.y + self.pole1_length)
        pole2_shape = pymunk.Segment(self.pole2_body, (0, 0), (0, self.pole2_length), 5)
        pole2_shape.friction = 0.0
        pole2_shape.filter = pymunk.ShapeFilter(group=1)

        static_body = self.space.static_body
        track_joint = pymunk.GrooveJoint(
            static_body, self.cart_body,
            (self.screen_width / 2 - self.track_limit, self.screen_height - self.cart_y),
            (self.screen_width / 2 + self.track_limit, self.screen_height - self.cart_y),
            (0, 0)
        )

        joint1 = pymunk.PivotJoint(
            self.cart_body, self.pole1_body,
            self.cart_body.position
        )
        joint1.collide_bodies = False

        joint2 = pymunk.PivotJoint(
            self.pole1_body, self.pole2_body,
            (self.cart_body.position.x, self.cart_body.position.y + self.pole1_length)
        )
        joint2.collide_bodies = False

        self.space.add(
            self.cart_body, cart_shape,
            self.pole1_body, pole1_shape,
            self.pole2_body, pole2_shape,
            track_joint, joint1, joint2
        )

    def _get_observation(self):
        cart_x = self.cart_body.position.x
        cart_center = self.screen_width / 2
        norm_cart_x = (cart_x - cart_center) / self.track_limit
        norm_cart_vx = np.clip(self.cart_body.velocity.x / 500.0, -5.0, 5.0)

        angle1 = self.pole1_body.angle
        angle1 = ((angle1 + math.pi) % (2 * math.pi)) - math.pi
        omega1 = np.clip(self.pole1_body.angular_velocity / math.pi, -10.0, 10.0)

        angle2 = self.pole2_body.angle - self.pole1_body.angle
        angle2 = ((angle2 + math.pi) % (2 * math.pi)) - math.pi
        omega2 = np.clip(self.pole2_body.angular_velocity / math.pi, -10.0, 10.0)

        return np.array([
            norm_cart_x,
            norm_cart_vx,
            angle1,
            omega1,
            angle2,
            omega2
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.space is not None:
            shapes = list(self.space.shapes)
            for shape in shapes:
                self.space.remove(shape)
            constraints = list(self.space.constraints)
            for constraint in constraints:
                self.space.remove(constraint)
            bodies = list(self.space.bodies)
            for body in bodies:
                if body != self.space.static_body:
                    self.space.remove(body)
        self._create_space()
        self.current_step = 0

        noise = 0.05
        self.pole1_body.angle = self.np_random.uniform(-noise, noise)
        self.pole2_body.angle = self.np_random.uniform(-noise, noise)

        return self._get_observation(), {}

    def step(self, action):
        action_value = float(np.clip(action[0], -1.0, 1.0))
        force = action_value * self.force_magnitude
        self.cart_body.apply_force_at_local_point((force, 0), (0, 0))

        self.space.step(self.dt)
        self.current_step += 1

        obs = self._get_observation()
        angle1 = obs[2]
        omega1 = obs[3]
        angle2 = obs[4]
        omega2 = obs[5]
        cart_x = obs[0]
        cart_vx = obs[1]

        reward = self._compute_reward(angle1, omega1, angle2, omega2, cart_x, action_value)

        terminated = bool(
            abs(angle1) > math.pi * 0.6 or
            abs(angle2) > math.pi * 0.6 or
            abs(cart_x) > 0.95
        )
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, angle1, omega1, angle2, omega2, cart_x, action):
        if self.reward_type == "baseline":
            reward = math.cos(angle1) + math.cos(angle2)
        else:
            upright = math.cos(angle1) + math.cos(angle2)
            center_penalty = -abs(cart_x) * 0.1
            velocity_penalty = -(abs(omega1) + abs(omega2)) * 0.01
            action_penalty = -(action ** 2) * 0.001
            reward = upright + center_penalty + velocity_penalty + action_penalty
        return float(reward)

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Double Inverted Pendulum")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((240, 240, 240))

        pygame.draw.line(
            self.screen,
            (100, 100, 100),
            (int(self.screen_width / 2 - self.track_limit), int(self.screen_height - self.cart_y)),
            (int(self.screen_width / 2 + self.track_limit), int(self.screen_height - self.cart_y)),
            4
        )

        cart_pos = self.cart_body.position
        cart_rect = pygame.Rect(
            int(cart_pos.x) - 40,
            int(self.screen_height - cart_pos.y) - 10,
            80, 20
        )
        pygame.draw.rect(self.screen, (50, 100, 200), cart_rect, border_radius=4)

        pole1_start = (int(cart_pos.x), int(self.screen_height - cart_pos.y))
        pole1_angle = self.pole1_body.angle
        pole1_end = (
            int(pole1_start[0] + self.pole1_length * math.sin(pole1_angle)),
            int(pole1_start[1] - self.pole1_length * math.cos(pole1_angle))
        )
        pygame.draw.line(self.screen, (200, 80, 50), pole1_start, pole1_end, 10)
        pygame.draw.circle(self.screen, (180, 60, 30), pole1_start, 8)

        pole2_start = pole1_end
        pole2_angle = self.pole2_body.angle
        pole2_end = (
            int(pole2_start[0] + self.pole2_length * math.sin(pole2_angle)),
            int(pole2_start[1] - self.pole2_length * math.cos(pole2_angle))
        )
        pygame.draw.line(self.screen, (50, 180, 80), pole2_start, pole2_end, 8)
        pygame.draw.circle(self.screen, (30, 160, 60), pole2_start, 6)
        pygame.draw.circle(self.screen, (255, 200, 0), pole2_end, 6)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None