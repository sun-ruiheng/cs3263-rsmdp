import random


class TradingTester:
    @staticmethod
    def test(env, policy, time_frame, n_episodes):
        #init accumulators
        result = []
        history_exec = []

        for e in range(n_episodes):
            #execute same scenario
            rand_probs = []
            for _ in range(time_frame):
                rand_probs.append(random.uniform(0, 1))

            s0 = 0
            total_reward = 0
            s = s0
            env.reset()
            for t in range(time_frame):
                print(f"Episode: {e+1}/{n_episodes}, Time frame: {t+1}/{time_frame}", end="\r", flush=True)
                a = policy[s]
                s_next, r = env.execute_with_prob(s, a, rand_probs[t])
                total_reward += r
                s = s_next

            result.append(total_reward)
            history_exec.append(env.history_states)

        print("RESULT")
        print(result)
        print()

        return result, history_exec