# Default Value Iteration method
import numpy as np


class ValueIteration:
    @staticmethod
    def run(env, gamma=1, epsilon=1e-8, quiet=False):
        V = np.zeros(len(env.S))
        policy = np.zeros(len(env.S))
        steps = 0
        updates = 0
        while True:
            steps += 1
            prev_V = np.copy(V)
            for s in env.S:
                q, q_updates = ValueIteration.action_value(env, V, s, gamma)
                V[s] = max(q)
                policy[s] = np.argmax(q)
                updates += q_updates

            delta = np.max(np.fabs(prev_V - V))
            if not quiet:
                print('{} {}'.format(steps, delta), end="\r", flush=False)
            if delta < epsilon:
                break
        return policy, V, steps, updates

    @staticmethod
    def action_value(env, V, s, gamma=1):
        q = np.zeros(len(env.A))
        q_updates = 0
        for a in env.A:
            for s_next, t, r in env.P[(s, a)]:
                q_updates += 1
                q[a] += t * (r + gamma * V[s_next])

        return q, q_updates

# Value itaration with exponential lambda
class ValueIterationExp:
    @staticmethod
    def utility_function(x, lamb):
        return -1 * np.sign(lamb) * np.exp(-lamb * x)


    @staticmethod
    def run(env, lamb, gamma, epsilon=1e-8, quiet=False):
        V = np.zeros(len(env.S))
        policy = np.zeros(len(env.S))
        steps = 0
        updates = 0
        while True:
            steps += 1
            
            prev_V = np.copy(V)
            for s in env.S:
                q, q_updates = ValueIterationExp.action_value_rs(env, V, s, lamb, gamma)

                V[s] = (np.log(-1 * np.sign(lamb) * max(q)) / -lamb)

                policy[s] = np.argmax(q)
                updates += q_updates

            delta = np.max(np.fabs(prev_V - V))
            if not quiet:
                print('{} {}'.format(steps, delta), end="\r", flush=True)
            if delta < epsilon:
                break
        return policy, V, steps, updates

    @staticmethod
    def action_value_rs(env, V, s, lamb, gamma):
        q = np.zeros(len(env.A))
        q_updates = 0
        for a in env.A:
            for s_next, t, r in env.P[(s, a)]:
                q_updates += 1

                q[a] += t * ValueIterationExp.utility_function(r + gamma * V[s_next], lamb)

        return q, q_updates