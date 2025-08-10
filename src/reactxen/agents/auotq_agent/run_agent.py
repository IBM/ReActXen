from reactxen.agents.auotq_agent.agent import RecipeAgent

WindTurbineAgent = RecipeAgent(name="Maximo")
WindTurbineAgent.set_asset_class("Wind Turbine")
WindTurbineAgent.init_chat(round=1)