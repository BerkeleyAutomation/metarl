"""RC-Car domain"""

from rlpy.Tools import plt, bound, wrap, mpatches, id2vec
import matplotlib as mpl
from rlpy.Domains.Domain import Domain
import numpy as np
from RCCarModified import RCCarModified

__author__ = "Alborz Geramifard"


# _fig = plt.gcf()

class RCCarBarriers(RCCarModified):

    """
    This has logic for replacing the dynamics and also putting obstacles

    **STATE:** 4 continuous dimensions:

    * x, y: (center point on the line connecting the back wheels),
    * speed (S on the webpage)
    * heading (theta on the webpage) w.r.t. body frame.
        positive values => turning right, negative values => turning left

    **ACTIONS:** Two action dimensions:

    * accel [forward, coast, backward]
    * phi [turn left, straight, turn Right]

    This leads to 3 x 3 = 9 possible actions.

    **REWARD:** -1 per step, 100 at goal.

    **REFERENCE:**

    .. seealso::
        http://planning.cs.uiuc.edu/node658.html

    """
    gcf = None
    actions_num = 9
    state_space_dims = 4
    continuous_dims = np.arange(state_space_dims)

    ROOM_WIDTH = 5  # in meters
    ROOM_HEIGHT = 4  # in meters
    XMIN = -ROOM_WIDTH / 2.0
    XMAX = ROOM_WIDTH / 2.0
    YMIN = -ROOM_HEIGHT / 2.0
    YMAX = ROOM_HEIGHT / 2.0
    ACCELERATION = .1
    TURN_ANGLE = np.pi / 5
    SPEEDMIN = -.3
    SPEEDMAX = .3
    HEADINGMIN = -np.pi
    HEADINGMAX = np.pi
    INIT_STATE = np.array([-2.0, -0.0, 0.0, 0.])
    GOAL = [1., 0.]
    GOAL_RADIUS = 0.4
    # INIT_STATE = np.array([0, 0, 0.0, 2.])
    episodeCap = 200
    GOAL_REWARD = 500.

    GRASS_SPEED = 0.05
    
    actions = np.outer([-1, 0, 1], [-1, 0, 1])


    def __init__(self, angle=None, wall_array=[(-0.5,-2, 0.2, 2.5)], **kwargs):
        self.environment_angle = angle

        if wall_array is not None:
            for elem in wall_array: # (left corner) x, y, dx, dy
                assert len(elem) == 4
            self.wall_array = np.array(wall_array)
        super(RCCarBarriers, self).__init__(**kwargs)

    def step(self, a):
        if self.random_state.random_sample() < self.noise:
            # Random Move
            self.slips.append(self.state[:2])
            a = self.random_state.choice(self.possibleActions())
        # Map a number between [0,8] to a pair. The first element is
        # acceleration direction. The second one is the indicator for the wheel
        # a = 4
        acc, turn = id2vec(a, [3, 3])
        acc -= 1                # Mapping acc to [-1, 0 1]
        turn -= 1                # Mapping turn to [-1, 0 1]

        # Calculate next state
        ns = self._action_dynamics(self.state, acc, turn)


        self.state = ns.copy()
        terminal = self.isTerminal() or self._bumped(ns)
        # quad_reward = -(np.linalg.norm(self.state[0:2] - self.GOAL))

        r = self._reward(ns, terminal)

        return r, ns, terminal, self.possibleActions()

    def _reward(self, ns, terminal):
        r = self.GOAL_REWARD if terminal else self.STEP_REWARD

        # Collision to wall => set reward to bad
        if self._bumped(ns) or ns[0] == self.XMIN or ns[0] == self.XMAX or ns[1] == self.YMIN or ns[1] == self.YMAX:
            r = -100
        return r

    def _bumped(self, state):
        if self.wall_array is None:
            return False
        # assumes the box will be big enough such that it won't be skipped
        x, y = state[:2]
        for wallx, wally, dx, dy in self.wall_array:
            left =  wallx - self.CAR_WIDTH
            right =  wallx + dx + self.CAR_WIDTH
            bottom =  wally - self.CAR_WIDTH
            top =  wally + dy + self.CAR_WIDTH

            if ((left <= x <= right) and
                (bottom <= y <= top)):
                return True
        return False



    def _action_dynamics(self, state, acc, turn):
        # x, y, speed, heading = state
        if self.environment_angle is not None:
            print "NEUTRAL ENVIRONMENT"
            # # return self._dynamics_slope(state, acc, turn, angle=self.environment_angle)
            # return self._dynamics_slide_right(state, acc, turn)
        return self._dynamics_regular(state, acc, turn)

    def _check_value_bounds(self, nx, ny, nspeed, nheading):
        nx = bound(nx, self.XMIN, self.XMAX)
        ny = bound(ny, self.YMIN, self.YMAX)
        nspeed = bound(nspeed, self.SPEEDMIN, self.SPEEDMAX)
        nheading = wrap(nheading, self.HEADINGMIN, self.HEADINGMAX)
        ns = np.array([nx, ny, nspeed, nheading])
        return ns

    def _dynamics_regular(self, state, acc, turn):
        x, y, speed, heading = state
        nx = x + speed * np.cos(heading) * self.delta_t
        ny = y + speed * np.sin(heading) * self.delta_t
        nspeed = speed + acc * self.ACCELERATION * self.delta_t
        nheading    = heading + speed / self.CAR_LENGTH * \
            np.tan(turn * self.TURN_ANGLE) * self.delta_t

        ns = self._check_value_bounds(nx, ny, nspeed, nheading)
        return ns

    def _dynamics_slope(self, state, acc, turn, angle=np.pi/2):
        x, y, speed, heading = state

        nx = x + speed * np.cos(heading) * self.delta_t 
        ny = y + speed * np.sin(heading) * self.delta_t
        nspeed = speed + acc * (self.ACCELERATION) * self.delta_t  \
                            - (0.01 * np.cos(heading + angle) * abs(np.cos(heading + angle))) #SLOPE 
        nheading    = heading + speed / self.CAR_LENGTH * \
            np.tan(turn * self.TURN_ANGLE) * self.delta_t

        # Bound values
        ns = self._check_value_bounds(nx, ny, nspeed, nheading)
        return ns

    def _dynamics_slide(self, state, acc, turn, bias=0.):
        x, y, speed, heading = state
        nx = x + speed * np.cos(heading) * self.delta_t + bias
        ny = y + speed * np.sin(heading) * self.delta_t
        nspeed = speed + acc * self.ACCELERATION * self.delta_t
        nheading    = heading + speed / self.CAR_LENGTH * \
            np.tan(turn * self.TURN_ANGLE) * self.delta_t

        ns = self._check_value_bounds(nx, ny, nspeed, nheading)
        return ns

    def _dynamics_slide_left(self, state, acc, turn):
        assert False, "Method DEPRECATED"

    def _dynamics_slide_right(self, state, acc, turn):
        assert False, "Method DEPRECATED"

    def _dynamics_turn(self, state, acc, turn, bias=0.):
        x, y, speed, heading = state
        nx = x + speed * np.cos(heading) * self.delta_t
        ny = y + speed * np.sin(heading) * self.delta_t
        nspeed = speed + acc * self.ACCELERATION * self.delta_t
        nheading    = heading + speed / self.CAR_LENGTH * \
            np.tan(turn * self.TURN_ANGLE) * self.delta_t + bias

        ns = self._check_value_bounds(nx, ny, nspeed, nheading)
        return ns

#     def _dynamics_slipping(self, state, acc, turn):
#         x, y, speed, heading = state

#         nx = x + speed * np.cos(heading) * self.delta_t
#         ny = y + speed * np.sin(heading) * self.delta_t
#         nspeed = speed + acc * (self.ACCELERATION) * self.delta_t 
#         nheading    = heading # + speed / self.CAR_LENGTH * \
# #            np.tan(turn * self.TURN_ANGLE) * self.delta_t

#         # Bound values
#         ns = self._check_value_bounds(nx, ny, nspeed, nheading)
#         return ns

    # def _grass_step(self, speed):
    #     if speed - self.ACCELERATION > self.GRASS_SPEED * 2:
    #         ns = speed - self.ACCELERATION * 2

    #     return ns
    # def _normal_step(self, a):
    #     nx = x + speed * np.cos(heading) * self.delta_t
    #     ny = y + speed * np.sin(heading) * self.delta_t
    #     nspeed = speed + acc * self.ACCELERATION * self.delta_t
    #     nheading    = heading + speed / self.CAR_LENGTH * \
    #         np.tan(turn * self.TURN_ANGLE) * self.delta_t

    #     # Bound values
    #     nx = bound(nx, self.XMIN, self.XMAX)
    #     ny = bound(ny, self.YMIN, self.YMAX)
    #     nspeed = bound(nspeed, self.SPEEDMIN, self.SPEEDMAX)
    #     nheading = wrap(nheading, self.HEADINGMIN, self.HEADINGMAX)

    #     ns = np.array([nx, ny, nspeed, nheading])
    #     self.state = ns.copy()
    #     terminal = self.isTerminal()
    #     r = self.GOAL_REWARD if terminal else self.STEP_REWARD

    #     # if r == self.GOAL_REWARD:
    #     #     print "SUCCESS - Reached goal"

    #     # Collision to wall => set the speed to zero
    #     if nx == self.XMIN or nx == self.XMAX or ny == self.YMIN or ny == self.YMAX:
    #         r = -100


    #     return r, ns, terminal, self.possibleActions()

    # def 
   
    def showDomain(self, a):
        if self.gcf is None:
            self.gcf = plt.gcf()

        s = self.state
        # Plot the car
        x, y, speed, heading = s
        car_xmin = x - self.REAR_WHEEL_RELATIVE_LOC
        car_ymin = y - self.CAR_WIDTH / 2.
        if self.domain_fig is None:  # Need to initialize the figure
            self.domain_fig = plt.figure()

            if self.wall_array is not None:
                for xmin, ymin, dx, dy in self.wall_array:
                    plt.gca().add_patch(
                        mpatches.Rectangle(
                            [xmin,
                             ymin],
                            dx,
                            dy,
                            color='c',
                            alpha=.4)
                    )
            # Goal
            plt.gca(
            ).add_patch(
                plt.Circle(
                    self.GOAL,
                    radius=self.GOAL_RADIUS,
                    color='g',
                    alpha=.4))
            plt.xlim([self.XMIN, self.XMAX])
            plt.ylim([self.YMIN, self.YMAX])
            plt.gca().set_aspect('1')
        # Car
        if self.car_fig is not None:
            plt.gca().patches.remove(self.car_fig)

        if self.slips:            
            slip_x, slip_y = zip(*self.slips)
            try:
                line = plt.axes().lines[0]
                if len(line.get_xdata()) != len(slip_x): # if plot has discrepancy from data
                    line.set_xdata(slip_x)
                    line.set_ydata(slip_y)
            except IndexError:
                plt.plot(slip_x, slip_y, 'x', color='b')

        self.car_fig = mpatches.Rectangle(
            [car_xmin,
             car_ymin],
            self.CAR_LENGTH,
            self.CAR_WIDTH,
            alpha=.4)
        rotation = mpl.transforms.Affine2D().rotate_deg_around(
            x, y, heading * 180 / np.pi) + plt.gca().transData
        self.car_fig.set_transform(rotation)
        plt.gca().add_patch(self.car_fig)

        plt.draw()
        # self.gcf.canvas.draw()
        plt.pause(0.001)
