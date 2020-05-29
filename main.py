#!/usr/bin/python3
import os
import argparse
from scripts.engines.server_manager import ServerManagerBinary
from scripts.engines.setup_world import SetupWorld
from scripts.rl_agent.ddpg_agent import ddpgAgent
from scripts.rl_agent.input_preprocessor import InputPreprocessor
import numpy as np
import sys
import linecache
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assurance Monitoring for RL-based emergency braking system.')
    parser.add_argument("-g", "--gui", help="set gui mode.", action="store_true")
    parser.add_argument("-t", "--testing", help="set testing mode", action="store_true", default=False)
    parser.add_argument("-e", "--episode", help="set the number of episode", type=int, default=1)

    args = parser.parse_args()

    try:
        carla_server = ServerManagerBinary({'CARLA_SERVER': os.environ["CARLA_SERVER"]})
        carla_server.reset()
        carla_server.wait_until_ready()
        env = SetupWorld(town=1, gui=args.gui)
        agent = ddpgAgent(Testing=args.testing)
        input_preprocessor = InputPreprocessor()
        step = 0
        if args.testing is False:
            writer = SummaryWriter()

        for episode in range(args.episode):
            initial_distance = np.random.normal(100, 1)
            initial_speed = np.random.uniform(26,30)
            s = env.reset(initial_distance, initial_speed)
            print("Episode {} is starting...".format(episode))
            s = input_preprocessor(s)
            epsilon = 1.0 - (episode+1)/(args.episode)
            while True:
                a = agent.getAction(s, epsilon)
                s_, r, done= env.step(a[0][0])
                s_ = input_preprocessor(s_)
                if args.testing is False:
                    agent.storeTrajectory(s, a, r, s_, done)
                    critic_loss = agent.learn()
                    writer.add_scalar('scalar/cirtic_loss', critic_loss, step)
                s = s_
                step += 1
                if done:
                    print("Episode {} is done, the reward is {}".format(episode,r))
                    if args.testing is False:
                        writer.add_scalar('scalar/reward', r, episode)
                    break

            if args.testing is False:
                if np.mod(episode, 10) == 0:
                    agent.save_model()
        carla_server.stop()
        if args.testing is False:
            writer.close()
    
    except AssertionError as error:
        print(repr(error))
    except Exception as error:
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print ('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
        #print('Caught this error: ' + repr(error))
        carla_server.stop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        carla_server.stop()

