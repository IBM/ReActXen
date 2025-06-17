MPE_SIMPLE4 = """Question: Is there any failure mode for Gearbox 3 relevant to the sensor power input?
Thought 1: I need to get the failure modes for Gearbox 3 to check if any of them are related to the sensor power input.
Action 1: Get Failure Mode Information
Action Input 1: Gearbox 3
Observation 1: Failure modes: ['(1) Bearing Failure: Caused by contamination, improper lubrication','(2) Gear Teeth Damage: Resulting from misalignment, excessive load', '(3) Shaft Misalignment: Due to installation errors, wear and tear', '(4) Lubrication System Failure: Resulting from contamination, lack of lubrication'].
Thought 2: I need to check if any of the failure modes are related to the sensor power input. let me get the list of available sensors first.
Action 2: Get Available Sensors Information
Action Input 2: Gearbox 3
Observation 2: Available sensors: ["Gearbox 3 Input Speed","Gearbox 3 Output Speed","Gearbox 3 Temperature","Gearbox 3 Vibration","Gearbox 3 Oil Level","Gearbox 3 Power Input","Gearbox 3 Load"].
Thought 3: I now know sensor power input is in the list of available sensors and has obtained failure modes for Gearbox 3. I can check if any of the failure modes are related to the sensor power input and get the final answer.
Action 3: Finish
Action Input 3: Yes, there is a failure mode relevant to the sensor power input, specifically '(4) Lubrication System Failure: Resulting from contamination, lack of lubrication', which is related to the 'Gearbox 3 Power Input' sensor.

Question: Is there any anomaly can be detected using Gearbox 3 sensor data?
Thought 1: I need to get the sensor data for Gearbox 3 to check for anomalies.
Action 1: Get Sensor Data
Action Input 1: Gearbox 3
Observation 1: Sensor Data for Gearbox 3 is downloaded from SkySpark System as Gearbox 3-resource-1.
Thought 2: Now that I have the sensor data Gearbox 3-resource-1, I need to run anomaly detection on it to check for any anomalies.
Action 2: Get Anomaly Detection
Action Input 2: Gearbox 3-resource-1
Observation 2: All the input signal looks Okay. No Anomaly.
Thought 3: I now know the final answer. I should finish the task.
Action 3: Finish
Action Input 3: Based on received sensor data, there is no anomaleous behavious in Gearbox 3

Question: What are the failure modes for Chiller 6 that can be monitored using the available sensors?
Thought 1: I need to get the failure modes for Chiller 6 and then check which of them can be monitored using the available sensors.
Action 1: Get Failure Mode Information
Action Input 1: Chiller 6
Observation 1: Failure modes: ['(1) Compressor Overheating: Failed due to Normal wear, overheating','(2) Heat Exchangers: Controls: Loose set screws, brittle or broken phenolics due to Normal wear', '(3) Heat Exchangers: Fans: Degraded motor or worn bearing due to Overgreasing, lack of lubrication causing increased power use']
Thought 2: Now that I have the failure modes for Chiller 6, I need to get the available sensors for Chiller 6 and check which of the failure modes can be monitored using these sensors.
Action 2: Get Available Sensors Information
Action Input 2: Chiller 6
Observation 2: Available sensors: ["Chiller 6 Condenser Water Return To Tower Temperature","Chiller 6 Chiller Efficiency","Chiller 6 Tonnage","Chiller 6 Supply Temperature","Chiller 6 Return Temperature","Chiller 6 Run Status","Chiller 6 Condenser Water Flow","Chiller 6 Schedule","Chiller 6 Power Input","Chiller 6 Chiller % Loaded","Chiller 6 Liquid Refrigerant Evaporator Temperature","Chiller 6 Setpoint Temperature"]
Thought 3: I now need to compare the failure modes with the available sensors and identify which failure modes can be monitored.
Action 3: Finish
Action Input 3: The failure modes "Compressor Overheating" and "Heat Exchangers: Fans: Degraded motor or worn bearing" can potentially be monitored using the "Chiller 6 Power Input" sensor. However, there doesn't seem to be a direct sensor to monitor "Heat Exchangers: Controls: Loose set screws, brittle or broken phenolics due to Normal wear".

Question: Help me generate a work order for Power Input Signal's performance, evaluate the validity, and whether a work order already exists. If a work order already exists do not raise a work order
Thought 1: I need to generate a work order description for Power Input Signal's performance.
Action 1: Get work order description
Action Input 1: Power Input Signal
Observation 1: ['KPI_Name': 'Power Input Signal', 'KPI_Description': 'Monitors ...', 'failure_location': 'Electrical Systems: Motor', 'degradation_mechanism': 'Motor winding faults', 'reasoning': 'An increase in power ...'].
Thought 2: I have generated a work order description for Power Input Signal's performance. Now, I need to evaluate the validity of this work order.
Action 2: LLM as a judge for work order selection
Action Input 2: Power Input Signal
Observation 2: 'w_index': 0, 'explanation': 'The first ...'.
Thought 3: I have generated a work order description for Power Input Signal's performance. Now, I need to evaluate the validity of this work order.
Action 3: Existing work order comparison
Action Input 3: Power Input Signal
Observation 3: There is no existing work order in system.
Thought 4: I now know the final answer. I should finish the task.
Action 4: Finish
Action Input 4: A valid work order description for Power Input Signal's performance has been generated, and it is unique, so a new work order should be raised.

Question: Help me to modify the following rule statement using real sensor name: If Gearbox 3 is not rotating and gearbox oil pressure is greater than 50 PSI, raise an alert. Only raise an alert if the condition is met for 2 hours. 
Thought 1: I need to get the sensor information for Gearbox 3 to check for possible modification in variable alignment.
Action 1: Get Available Sensors Information
Action Input 1: Gearbox 3
Observation 1: [Gearbox Speed, Gearbox Oil Pressure, Gearbox Temperature, Vibration Level, Gearbox Current Draw, Timestamp]
Thought 2: I should deal with timestamp column since rule mentioned time related condiction such as hours. I now know the final answer.
Action 2: Finish
Action Input 2: The modified statement is : If Gearbox 3's Gearbox Speed is equal to 0 and Gearbox Oil Pressure is greater than 50 PSI at a given Timestamp, raise an alert. Only raise an alert if the condition is met for 2 consecutive hours, based on the Timestamp."""


MPE_REFLECTIONS = """

Previous Trial:
Question: get sensor data for Chiller 10 and then detect anomaly if it exists?
Thought 1: I need to get the sensor data for Chiller 10 to check for anomalies.
Action 1: Get Sensor Data
Action Input 1: Chiller 10
Observation 1: Sensor Data for Chiller 10 is downloaded from SkySpark System as Chiller 10-resource-10
Thought 2: Now that I have the sensor data Chiller 10-resource-10, I need to run anomaly detection on it to check for any anomalies.
Action 2: Get Anomaly Detection
Action Input 2: Chiller 10-resource-10
Chiller 10 resource-10
Observation 2: Model is not available in Model Catalog
Thought 3: Since the model is not available in the Model Catalog, I need to fine-tune an anomaly model using the sensor data.
Action 3: Fine Tune Anomaly Model
Action Input 3: Chiller 10-resource-10
Observation 3: Fine-tune model is created for Chiller 10 and registered in Model Catalog for Future Usage
Thought 4: Now that I have fine-tuned the anomaly model, I can run anomaly detection on the sensor data again.
Action 4: Get Sensor Data
Action Input 4: Chiller 10
Observation 4: Sensor Data for Chiller 10 is downloaded from SkySpark System as Chiller 10-resource-10
Thought 5: Now that I have the fine-tuned model and the sensor data, I can run anomaly detection again to check for any anomalies.
Action 5: Get Sensor Data
Action Input 5: Chiller 10
Observation 5: Sensor Data for Chiller 10 is downloaded from SkySpark System as Chiller 10-resource-10
Thought 6: Now that I have the fine-tuned model and the sensor data, I can run anomaly detection again to check for any anomalies.
Action 6: Get Sensor Data
Action Input 6: Chiller 10
Observation 6: Sensor Data for Chiller 10 is downloaded from SkySpark System as Chiller 10-resource-10

Reflection: I got stuck in a loop where I kept trying to search 'Get Sensor Data' but this action will download same data multiple times. Instead, I should have tried 'Get Anomaly Detection' directly since data was already downloaded by Action 1.

Previous Trial:
Question: Have there been any issues with Chiller 8?
Thought 1: First, I need to get the sensor data for Chiller 8.
Action 1: Get Sensor Data
Action Input 1: Chiller 8
Observation 1: Sensor Data for Chiller 8 is downloaded from SkySpark System as Chiller 8-resource-0
Thought 2: Now that I have the sensor data for Chiller 8, I need to run anomaly detection on it to check for any anomalies.
Action 2: Get Anomaly Detection
Action Input 2: Chiller 8-resource-0
Observation 2: Anomaly is detected in Power Input Signal
Thought 3: Now that an anomaly has been detected in the Power Input Signal, I need to generate a work order for this anomaly.
Action 3: Get work order description
Action Input 3: Power Input Signal
Observation 3: Here is the work order description, power signal is very high.
Thought 4: Now that I have the work order descriptions, I will generate work order.
Action 4: Finish
Action Input 4: work order is generated.

Reflection: I realize that I went beyond the original question's scope by generating a work order, when I should have only reported the anomaly detection result."""

COTQA_SIMPLE4 = """
Question: Is there any failure mode for Gearbox 3 relevant to the sensor power input?
Thought: Let's think step by step. I need to obtain the failure modes for Gearbox 3 to assess their relevance to the sensor power input, then retrieve the available sensors for Gearbox 3 to cross-reference with the failure modes, and finally evaluate if any of the identified failure modes correlate with the sensor power input to reach a conclusive answer.
Action: Finish[Get Failure Mode[Gearbox 3], Get Available Sensors[Gearbox 3]]

Question: Is there any anomaly that can be detected using Gearbox 3 sensor data?
Thought: Let's think step by step. I need to obtain the sensor data for Gearbox 3 to check for anomalies, then run anomaly detection on the data retrieved from the SkySpark System as Gearbox 3-resource-1, and finally conclude whether there are any anomalies present in the Gearbox 3 sensor data.
Action: Finish[Get Sensor Data[Gearbox 3], Get Anomaly Detection[Gearbox 3-resource-1]]

Question: What are the failure modes for Chiller 6 that can be monitored using the available sensors?
Thought: Let's think step by step. First, I need to obtain the failure modes for Chiller 6 and then check which of them can be monitored using the available sensors. Next, I'll retrieve the available sensors for Chiller 6 and cross-reference them with the failure modes to identify any overlap where monitoring is possible.
Action: Finish[Get Failure Mode[Chiller 6], Get Available Sensors[Chiller 6]]

Question: Help me generate a work order for Power Input Signal's performance, evaluate its validity, and check if a work order already exists. If a work order already exists, do not raise a new one.
Thought: Let's think step by step. First, I need to generate a work order description for Power Input Signal's performance. Then, I'll evaluate the validity of this work order description. Next, I'll check if there is already an existing work order for Power Input Signal. Based on this assessment, I'll determine whether to raise a new work order.
Action: Finish[Get work order description[Power Input Signal], LLM as a judge for work order selection[Power Input Signal], Existing work order comparison[Power Input Signal]]"""

COT_SIMPLE_REFLECTION2 = """
Question: Get sensor data for Chiller 10 and then detect anomaly if it exists?
Thought: Let's think step by step. Get sensor data for Chiller 10, then run anomaly detection on Chiller 10 data; fine-tune the anomaly model using Chiller 10 data, followed by getting sensor data for Chiller 10 again three times consecutively.
Action: Finish[Get Sensor Data[Chiller 10], Get Anomaly Detection[Chiller 10-resource-10], Fine Tune Anomaly Model[Chiller 10-resource-10], [Get Sensor Data[Chiller 10], Get Sensor Data[Chiller 10], [Get Sensor Data[Chiller 10]]

Reflection: I got stuck in a loop where I kept trying to search 'Get Sensor Data' but this action will download same data multiple times. Instead, I should have tried 'Get Anomaly Detection' directly since data was already downloaded by Action.

Question: Have there been any issues with Chiller 8?
Thought 1: Let's think step by step. First, I need to get the sensor data for Chiller 8. I need to run anomaly detection followed by generate a work order.
Action: Finish[Get Sensor Data[Chiller 8], Get Anomaly Detection[Chiller 8-resource-0, Get work order description[Power Input Signal]]

Reflection: I realize that I went beyond the original question's scope by generating a work order, when I should have only reported the anomaly detection result.
"""

COT = """Relevant Context: .....
Question: ....?
Thought: ....
Action: Finish[...]

Relevant Context: ....
Question: ....?
Thought: ....
Action: Finish[...]

Relevant Context: ....
Question: ....?
Thought: ....
Action: Finish[...]
"""

COT_REFLECT = """
Relevant Context: ....
Question: ...?
Thought: ....
Action: Finish[...]

Reflection: My answer was correct based on the context, but may not be the exact answer stored by the grading environment. Next time, I should try to provide a less verbose answer like "Pulitzer Prize" or "Nobel Prize."

Context: ....
Question: ....?
Thought: ....
Action: Finish[...]

Reflection: Upon reflecting on the incorrect answer I provided, I realize that I may not have given the full ..."""

COT_REFLECT2 = """Relevant Context: ....
Question: ...?
Thought: ....
Action: Finish[...]

Reflection: I made an error in my response, as I incorrectly stated that ....

Relevant Context: ....
Question: ....?
Thought: ....
Action: Finish[...]

Reflection: My response was incorrect, as I stated that ....

Relevant Context: ....
Question: ...?
Thought: ....
Action: Finish[...]

Reflection: My answer was incorrect, as I stated that ....
"""

MetaPlanSample = """
Question: Decompose the task: Forecast temperature changes for a cooling system using data in "temperature_data.csv"
Thought: Think about how to break down the task into smaller, manageable steps using the available tools.
Final Answer: Here are the steps to decompose the task:
1. Query the available forecasting tool to retrieve suitable models for temperature forecasting.
2. Select an appropriate model based on the available data in "temperature_data.csv".
3. Perform time series forecasting on the dataset "temperature_data.csv" using the selected model.
4. Set the forecast horizon to 1 day and configure other relevant parameters.
5. Retrieve and display the forecasted temperature results.
6. Repeat steps 3-5 if more granular forecasts or adjustments are needed (this may involve multiple iterations).
7. Finish.
"""