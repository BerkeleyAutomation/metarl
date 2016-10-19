from rlpy.Agents import Q_Learning
from rlpy.Domains.Domain import Domain
import numpy as np
from copy import deepcopy
from collections import Counter, defaultdict
from itertools import combinations
from rlpy.Tools import plt, bound, wrap, mpatches, id2vec
from PolicyMixer import PolicyMixer
import matplotlib as mpl

# import rllab

class RCPolicyMixer(PolicyMixer):
    """Reimplements the visualization to get good stuff"""

    def __init__(self, *args):
    	super(RCPolicyMixer, self).__init__(*args)
        print "$$$$$$$ Initializing RCPolicyMixer"
        import time; time.sleep(1)
        self.colors = ['red', 'green']
    	assert hasattr(self.actual_domain, "TURN_ANGLE")

    def showDomain(self, a):

    	dom = self.actual_domain
        if dom.gcf is None:
            dom.gcf = plt.gcf()

        # Plot the car
        x, y, speed, heading = dom.state
        car_xmin = x - dom.REAR_WHEEL_RELATIVE_LOC
        car_ymin = y - dom.CAR_WIDTH / 2.
        if dom.domain_fig is None:  # Need to initialize the figure
            dom.domain_fig = plt.figure()
            if hasattr(dom, "wall_array" ) and dom.wall_array is not None:
                for xmin, ymin, dx, dy in dom.wall_array:
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
                    dom.GOAL,
                    radius=dom.GOAL_RADIUS,
                    color='g',
                    alpha=.4))
            plt.xlim([dom.XMIN, dom.XMAX])
            plt.ylim([dom.YMIN, dom.YMAX])
            plt.gca().set_aspect('1')
        # Car
        if dom.car_fig is not None:
            plt.gca().patches.remove(dom.car_fig)

        # if dom.slips:            
        #     slip_x, slip_y = zip(*dom.slips)
        #     try:
        #         line = plt.axes().lines[0]
        #         if len(line.get_xdata()) != len(slip_x): # if plot has discrepancy from data
        #             line.set_xdata(slip_x)
        #             line.set_ydata(slip_y)
        #     except IndexError:
        #         plt.plot(slip_x, slip_y, 'x', color='b')


        if self.curr_run:
            past = np.vstack([hist[0] for hist in self.curr_run])
            action_chosen = np.array([hist[1]  for hist in self.curr_run])
            # if len(self.curr_run) % 20 == 0:

            try:
                scat = plt.axes().collections[0]
                if len(scat.get_offsets()) != len(past): # if plot has discrepancy from data
                    scat.set_offsets(past[:, :2])
                    scat.set_array(action_chosen)
            except IndexError:
                scat = plt.scatter(past[:, 0], past[:, 1])
                scat.set_cmap('viridis')
                scat.set_clim(vmin=0, vmax=1)


        dom.car_fig = mpatches.Rectangle(
            [car_xmin,
             car_ymin],
            dom.CAR_LENGTH,
            dom.CAR_WIDTH,
            alpha=.4)
        rotation = mpl.transforms.Affine2D().rotate_deg_around(
            x, y, heading * 180 / np.pi) + plt.gca().transData
        dom.car_fig.set_transform(rotation)
        plt.gca().add_patch(dom.car_fig)

        plt.draw()
        # dom.gcf.canvas.draw()
        plt.pause(0.001)