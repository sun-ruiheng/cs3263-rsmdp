import math
import random
import numpy as np
import pandas as pd


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
                print('{} {}'.format(steps, delta), end="\r", flush=False)
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

class TradingEnvironment:
    def __init__(self, training_data, expected_return_step_size, round_num_digits, transaction_cost=0.002, bank=100, action_interval=25, action_total=100):
        n = len(training_data) + 1 # +1 for cash
        self.expected_return_step_size = expected_return_step_size
        self.round_num_digits = round_num_digits
        self.training_data = self._calculate_training_data(training_data)
        self.mean_rewards = self._calculate_rewards(self.training_data)
        self.transitions, self.transition_keys = self._calculate_transitions(self.training_data)
        self.transaction_cost = transaction_cost
        self.open_bank, self.current_bank = bank, bank
        self.real_data_index = self.expected_return_step_size - 1
        self.history_states = []
        self.actions, self.A = self._generate_actions(n, action_interval, action_total)
        self.states, self.S = self._generate_state(self.actions, self.transition_keys)
        self.P = self._calculate_probabilities(self.states, self.actions, self.transitions, self.transition_keys)        

    def _calculate_return(self, data):
        data['return'] = data['average'].pct_change()
        data['return'] = data['return'].fillna(float('nan'))
        data.loc[data['return'] == 0, 'return'] = 0

        return data

    def _calculate_expected_return(self, data):
        data['expected_return'] = data['return'].rolling(window=self.expected_return_step_size).mean().round(self.round_num_digits)
        data['expected_return'] = data['expected_return'].fillna(float('nan'))
        data.loc[data['expected_return'] == 0, 'expected_return'] = 0

        return data
        
    def _calculate_training_data(self, data):
        training_data = []
        for d in data:
            d = d[d.average > 0].copy()
            d = self._calculate_return(d)
            d = self._calculate_expected_return(d)
            training_data.append(d)

        return training_data

    def _calculate_rewards(self, data):
        all_rewards = []
        for train in data:
            valid_data = train.dropna(subset=['expected_return'])
            grouped = valid_data.groupby(valid_data['expected_return'].astype(str))['return'].mean()
            mean_rewards = grouped.to_dict()
            all_rewards.append(mean_rewards)
            
        return all_rewards

    def _calculate_transitions(self, data):
        def calc(d):
            valid_data = d.dropna(subset=['expected_return'])
            last_expected_return = valid_data['expected_return'].iloc[-1]
            value_counts = valid_data['expected_return'].value_counts()

            weights = {}
            for key, count in value_counts.items():
                if key == last_expected_return and count > 1:
                    weights[str(key)] = 1/(count-1)
                else:
                    weights[str(key)] = 1/count

            pairs = pd.DataFrame({
                'current': valid_data['expected_return'].iloc[:-1].values,
                'next': valid_data['expected_return'].iloc[1:].values
            })
            pairs['current_str'] = pairs['current'].astype(str)
            pairs['next_str'] = pairs['next'].astype(str)

            transitions_temp = {}
            for current, group in pairs.groupby('current_str'):
                transitions_temp[current] = {}
                for _, row in group.iterrows():
                    next_val = row['next_str']
                    if next_val not in transitions_temp[current]:
                        transitions_temp[current][next_val] = 0
                    transitions_temp[current][next_val] += weights[current]
            transitions = {}

            for t in transitions_temp:
                transitions[t] = []
                for t_next, prob in transitions_temp[t].items():
                    transitions[t].append((t_next, prob))
                    
            return transitions
        
        transitions = []
        transition_keys = []
        for d in data:
            transition = calc(d)
            transitions.append(transition)
            transition_keys.append(sorted(list(transition.keys())))

        return transitions, transition_keys

    def _generate_actions(self, n, interval, total):
        # Generate all possible actions, then filter to keep only tuples that sum to total (100) ((legal actions))
        actions = pd.MultiIndex.from_product([list(range(0, total + interval, interval))] * n)
        actions = actions[actions.map(sum) == total]

        return list(actions), list(range(len(actions)))

    def _generate_state(self, actions, transition_keys):
        keys = pd.MultiIndex.from_product(transition_keys + [[0]])
        action_states = pd.MultiIndex.from_product([actions, list(keys)])

        states = []
        for (x, y) in action_states:
            # Zip the elements of x and y together and create tuples
            state_tuple = tuple(zip(x, y))
            states.append(state_tuple)

        return states, list(range(len(states)))
    
    def _calculate_probabilities(self, states, actions, transitions, transition_keys):
        P = {}
        total = len(states) * len(actions)
        count = 0
        lol = [0]
        
        def process_next_asset(lol, state_idx, action, current_state_data, asset_idx=0, current_prob=1.0, filtered_exps=None):
            if filtered_exps is None:
                filtered_exps = []
                
            if asset_idx >= len(state_data) - 1: # to exclude cash
                next_state_exps = [exp for exp, _ in filtered_exps]
                next_state = self._compose_state(action, *next_state_exps)
                prob = current_prob
                reward = self._calculate_reward(state_idx, next_state)
                P[(state_idx, a_idx)].append((next_state, prob, reward))

                return
                
            asset_state = state_data[asset_idx]
            next_returns = transitions[asset_idx][asset_state[1]]
            
            for next_exp, trans_prob in next_returns:
                if next_exp in transition_keys[asset_idx]:
                    new_prob = current_prob * trans_prob
                    new_filtered_exps = filtered_exps + [(next_exp, trans_prob)]
                    process_next_asset(lol,state_idx, action, state_data, 
                                    asset_idx + 1, new_prob, new_filtered_exps)
                    
        for s_idx, state_data in enumerate(states):
            for a_idx, action_data in enumerate(actions):
                P[(s_idx, a_idx)] = []
                count += 1
                print('{}/{}'.format(count, total), end="\r", flush=True)
                
                process_next_asset(lol, s_idx, action_data, state_data)
        
        return P

    def _compose_state(self, action, *next_expec_values):
        state_components = []

        for i, next_expec in enumerate(next_expec_values):
            state_components.append((action[i], next_expec))
        state_components.append((action[-1], 0))
        state = tuple(state_components)

        for i, s in enumerate(self.states):
            if s == state:
                return i
                
        raise ValueError(f"Could not find state for action {action} and expected returns {next_expec_values}")

    def _calculate_reward(self, last_state_index, next_state_index):
        last_state = self.states[last_state_index]
        next_state = self.states[next_state_index]
        
        if last_state == next_state:
            return 0
        
        total_dif = 0
        differences = []
        
        for i in range(len(last_state) - 1):
            diff = next_state[i][0] - last_state[i][0]
            differences.append(diff)
            if diff > 0:
                total_dif += diff
        
        total_gain = 0
        for i in range(len(last_state) - 1):  # Skip cash
            asset_gain = (self.mean_rewards[i][last_state[i][1]] * self.open_bank) * (next_state[i][0] / 100.0)
            total_gain += asset_gain
        
        discount = self.open_bank * (total_dif / 100.0) * self.transaction_cost
        
        max_allow = 50
        if total_dif > max_allow:
            return -100
        
        if total_dif == max_allow:
            if all(diff == 0 for diff in differences):
                return -100
        
        return total_gain - discount
    
    def execute(self, s, a):
        random_number = random.uniform(0, 1)
        return self.execute_with_prob(s, a, random_number)
            
    def execute_with_prob(self, s, a, random_number):
        t_sum = 0.0
        for s_next, t, r in self.P[(s, a)]:
            t_sum += t
            if random_number <= t_sum:
                self.history_states.append(self.states[s])
                return s_next, r
    
    def reset(self):
        self.current_bank = self.open_bank
        self.real_data_index = self.expected_return_step_size - 1
        self.history_states = []


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
    

if __name__ == "__main__":
    import time
    
    data_aapl = pd.read_csv("data/20250402_AAPL.csv")
    data_amzn = pd.read_csv("data/20250402_AMZN.csv")
    data_tsla = pd.read_csv("data/20250402_TSLA.csv")

    data = [data_aapl, data_amzn, data_tsla]
    expected_return_step_size = 5
    round_num_digits = 3
    transaction_cost = 0.005

    start_time = time.time()
    env = TradingEnvironment(data, expected_return_step_size, round_num_digits, transaction_cost)
    env_time = time.time() - start_time
    print(f"Time to create trading environment: {env_time:.2f} seconds")

    lamb = 0.5
    gamma = 0.99
    epsilon = 1e-3
    
    start_time = time.time()
    policy, V, steps, updates = ValueIterationExp.run(env, lamb=lamb, gamma=gamma, epsilon=epsilon)
    vi_time = time.time() - start_time
    print(f"Time to run value iteration: {vi_time:.2f} seconds")

    n_episodes = 10
    time_frame = 100
    result, history_exec = TradingTester.test(env, policy, time_frame, n_episodes)
    print(result)
    print(history_exec)

    print("DONE")


