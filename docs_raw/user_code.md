# Custom Control Classes
One of the core features of EMBRS is the ability to import custom control classes that can intervene and interact with the fire in a closed-loop manner. The custom class receives the entire simulation's state following each time-step and has the ability to perform an array of operations that essentially allow it to control the state of every discrete cell in the simulation. The integration of custom control classes is designed to be as simple and flexible as possible, there are just two requirements for compatibility.

## Compatibility Requirements

### Base Class Implementation

- Custom classes must implement the provided abstract base class called [`ControlClass`](./_autosummary/embrs.base_classes.control_base.rst) in `embrs/base_classes/control_base.py`.
- The base class requires the implementation of the `process_state` method. This is the method that is called after each iteration of an EMBRS simulation. It must take only a `FireSim` object as an input and shouldn't return anything.
- The implementation of the body of this method is completely up to the user, but it acts as the 'bridge' between the custom control class and the EMBRS simulation.


```python
# Abstract base class for custom control classes:

from abc import ABC, abstractmethod
from embrs.fire_simulator.fire import FireSim

class ControlClass(ABC):
    @abstractmethod
    def process_state(self, fire:FireSim):
        # This method must be implemented by custom classes

```

### Constructor Argument
- The constructor of any custom class should accept only one argument: a `FireSim` object.
- The custom class constructor is called automatically by the simulation which necessitates this requirement.

```python
# Example of a custom class constructor:

class ExampleCustomClass(ControlClass):
    def __init__(self, fire:FireSim):
        ...
```

(user_code:agents)=
## Agents
- EMBRS provides basic functionality for 'registering' agents with the simulation. This allows the simulation to automatically log the locations of agents during the course of a simulation so they can be visualized during real-time visualization and visualization in post.
- If you wish to have the simulation track your agents you must do the following:

    - **Implement AgentBase Class**
        - Your agent class must be a subclass of the provided [`AgentBase`](./_autosummary/embrs.base_classes.agent_base.rst) class located in 'base_classes/agent_base.py'. 
        - The base class just contains a constructor and a function `to_log_format()` that is called by the sim to log data about the agent as the simulation progression.
        - Sample code for implementing a custom agent class can be found [here.](examples:sample_agent)

        ```python
        # AgentBase Class:

        class AgentBase:
            def __init__(self, id:any, x: float, y: float, label:str=None,
                         marker:str='*', color:str='magenta'):
                self.id = id         # unique identifier for the agent
                self.x = x           # x position of the agent in meters
                self.y = y           # y position of the agent in meters
                self.label = label   # label for the agent in visualizations
                self.marker = marker # matplotlib marker for agent in visualizations
                self.color = color   # color of the marker in visualizations
            
            def to_log_format(self):
                ...

        ```
        
        ```{note}
      When updating the position of your agent in your code, change the value of x and y to do so.
        ```

    - **Add Agents to Sim**
        - Once your agent is constructed you must add your agent to the simulation
        - The `FireSim` class has a public method called `add_agent()` to do this.
        - Below is a simple example of how to construct an agent and add it to the sim:

        **Example of Adding Agent to Sim**
        ```python
        def process_state(self, fire:FireSim):
            ...            
            
            a_x = np.random.random() * fire.x_lim
            a_y = np.random.random() * fire.y_lim
            
            # SampleAgent would be replaced with your custom agent class
            agent = SampleAgent(0, a_x, a_y, fire, label = 'agent_0')

            fire.add_agent(agent) # Add agent to the sim

            ...

        ```

## Available Operations
- Refer to ['Fire Interface'](./interface_reference.md) to see all the operations custom classes can carry out on a `FireSim` instance.

## Sample Custom Classes
Two sample custom class implementations are provided with EMBRS. These are simple custom classes developed for the purpose of demonstrating the ability to perform suppression operations in response to the state of the fire in real-time.


### Prescribed Burning
- In the prescribed burning sample class, ten agents work with one another to carry out prescribed burns in order to reduce the fuel ahead of the fire. The agents leverage knowledge of the wind and the locations of control lines to determine the best places to start prescribed burns.
- This example can be found in ['examples/vi_controlled_burning.py'](examples:controlled_burning).

### Fuel Soaking
- In the fuel soaking sample class, ten agents work with one another to divide the fire front and increase the fuel moisture along the front as it evolves. This simulates dropping water ahead of a fire front in an attempt to suppress it.
- This example can be found in ['examples/vii_suppression.py'](examples:suppression).
