from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import TrainState


class Deadlock:
    #Oldest deadlock_data, primarily for data-gathering
    #Return a bunch of data related to deadlocks
    def deadlock_data(env: RailEnv):
        unfinished_total = set() #Should work
        unfinished_no_deadlock = set()
        location_has_agent = {}

        for agent in env.agents:
            if agent.state.is_on_map_state():
                location_has_agent[tuple(agent.position)] = 1 #Gets position of each agent

        # Looks for any agent that can still move
        for handle in env.get_agent_handles():
            agent = env.agents[handle]
            if agent.state.is_off_map_state():
                agent_virtual_position = agent.initial_position
            elif agent.state.is_on_map_state():
                agent_virtual_position = agent.position
            else:
                continue

            possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
            orientation = agent.direction

            for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
                    if possible_transitions[branch_direction]:
                        new_position = get_new_position(agent_virtual_position, branch_direction)
                        if new_position not in location_has_agent.keys():
                            unfinished_no_deadlock.add(handle)
                            continue
                    else:
                        unfinished_total.add(handle)

        deadlocks = len(unfinished_total) - len(unfinished_no_deadlock)
        is_deadlock = deadlocks > 0
        deadlocked_agents_list = unfinished_total - unfinished_no_deadlock
        
        return len(unfinished_total), len(unfinished_no_deadlock), deadlocks, is_deadlock, deadlocked_agents_list









    #gammel navn: deadlock_check5.py
    #Return a bunch of data related to deadlocks, for the observation specifically
    def deadlock_data_obs(env: RailEnv):
        location_has_agent = {}
        agent_new_positions = {}
        agent_new_positions_copy = {}
        position_has_agent = {}
        tmp_deadlocked = set()
        same_next_position = set() #kun 1 ud af de 2 bliver tilføjet her.

        for agent in env.agents:
            if agent.state.is_on_map_state():
                location_has_agent[tuple(agent.position)] = agent.handle #Gets position of each agent

                #populate agent_current_position:
                position_has_agent[tuple(agent.position)] = agent.handle

                #populate agent_new_positions:
                possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)

                new_positions = [] #obs hvis der er flere end 1 mulig new position, så er der jo ikke deadlock. Meget usandsynligt hvert fald.
                for branch_direction in [(agent.direction + i) % 4 for i in range(-1, 3)]:
                    if possible_transitions[branch_direction]:
                        new_position = get_new_position(agent.position, branch_direction)
                        new_positions.append(new_position)
                agent_new_positions[agent.handle] = new_positions
        agent_new_positions_copy = agent_new_positions.copy()

        for agent_handle in position_has_agent.values():
            if len(agent_new_positions[agent_handle]) > 1:
                continue #obs hvis der er flere end 1 mulig new position, så er der jo ikke deadlock. Meget usandsynligt hvert fald.
            else: #og her kommer det smarte:
                next_position = agent_new_positions[agent_handle][0]
                if next_position in position_has_agent.keys():
                    #men der er kun deadlock hvis de kører mod hinanden:
                    other_agent_handle = position_has_agent[next_position]
                    agent = env.agents[agent_handle]
                    other_agent = env.agents[other_agent_handle]
                    if abs(agent.direction - other_agent.direction) == 2: #n=0,e=1,s=2,w=3. Så hvis den absolutte forskel er 2, så kører man modsat.
                        tmp_deadlocked.add(agent_handle) #der er deadlock!
                        tmp_deadlocked.add(other_agent_handle)
                        continue

                        #hvis other er på mappen                     #Hvis de vil bytte plads så er der jo deadlock:
                    if other_agent_handle in agent_new_positions and next_position == other_agent.position:
                        if agent_new_positions[other_agent_handle][0] == agent.position:
                            tmp_deadlocked.add(agent_handle) #der er deadlock!
                            tmp_deadlocked.add(other_agent_handle)
                            continue
                
                #vi fjerner den vi kigger på. Og hvis der er en anden der vil have samme next position, så kører de jo ind i hinanden.
                agent_new_positions.pop(agent_handle, None)
                if [next_position] in agent_new_positions.values():
                    other_agent = list(agent_new_positions.keys())[list(agent_new_positions.values()).index([next_position])] #https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary
                    tmp_deadlocked.add(agent_handle) #der er deadlock!
                    tmp_deadlocked.add(other_agent)

                    if other_agent not in same_next_position: #vi vil kun tilføje hver anden.
                        same_next_position.add(agent_handle)
                    continue

                
        
        for agent_handle in position_has_agent.values():
            if len(agent_new_positions_copy[agent_handle]) > 1:  #vi bruger copy fordi vi har poppet pairs forinden.
                continue #obs hvis der er flere end 1 mulig new position, så er der jo ikke deadlock. Meget usandsynligt hvert fald.
            else: #og her kommer det smarte:
                next_position = agent_new_positions_copy[agent_handle][0]
                if agent_handle in tmp_deadlocked: #vi gider ikke tjekke dem der allerede er deadlocked.
                    continue
                elif agent_handle in agent_new_positions_copy and len(agent_new_positions_copy[agent_handle]) > 1:
                    continue #obs hvis der er flere end 1 mulig new position, så er der jo ikke deadlock. Meget usandsynligt hvert fald.
                elif next_position in position_has_agent:
                        if position_has_agent[next_position] in tmp_deadlocked: #Hvis vores næste position er optaget af en agent der er deadlocked. Så bliver vi også.
                            tmp_deadlocked.add(agent_handle)

        return tmp_deadlocked, same_next_position





    #Used in evaluate.py
    #With trainstate_done
    def check_if_all_blocked(env):
        """
        Checks whether all the agents are blocked (full deadlock situation).
        In that case it is pointless to keep running inference as no agent will be able to move.
        :param env: current environment
        :return:
        """

        # First build a map of agents in each position
        location_has_agent = {}
        for agent in env.agents:
            if (agent.state == TrainState.DONE or agent.state.is_on_map_state()) and agent.position:
                location_has_agent[tuple(agent.position)] = 1

        # Looks for any agent that can still move
        for handle in env.get_agent_handles():
            agent = env.agents[handle]
            if agent.state == TrainState.READY_TO_DEPART:
                agent_virtual_position = agent.initial_position
            elif agent.state.is_on_map_state():
                agent_virtual_position = agent.position
            elif agent.state == TrainState.DONE:
                agent_virtual_position = agent.target
            else:
                continue

            possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
            orientation = agent.direction

            for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(agent_virtual_position, branch_direction)

                    if new_position not in location_has_agent:
                        return False

        # No agent can move at all: full deadlock!
        return True


