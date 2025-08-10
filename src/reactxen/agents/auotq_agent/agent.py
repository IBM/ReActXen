from reactxen.agents.auotq_agent.genai.Chat import GenAIChatClient
from reactxen.agents.auotq_agent.genai.Instruct import GenAIInstructClient
from collections import defaultdict
import mlflow
import random
from reactxen.agents.auotq_agent.genai.utils import (
    filter_and_sort_questions,
    filter_questions_using_reference,
    filter_questions_using_TTR,
    filter_questions_using_Flan,
)
import pandas as pd
from colorama import Fore, Style
import uuid
from collections import OrderedDict
import pandas as pd
import difflib
from reactxen.agents.auotq_agent.task_description import get_description
from dotenv import load_dotenv

# from genai.schema import (
#     DecodingMethod,
#     ModerationHAP,
#     ModerationParameters,
#     TextGenerationReturnOptions,
# )
import os

# have model specific configuration
# QA does not need longer context to generate
# Max token generation need to be adjusted per

import ray


@ray.remote
def generate_response(llm, question, experiment_id):
    llm_answer = llm.create(
        context=None, messages=question, experiment_id=experiment_id
    )
    return llm_answer


@ray.remote
def generate_chain_response(
    agent1, agent2, input_question, input_answer, experiment_id
):
    intermediate_result = f"Here is the pair of question and answer: \n Question: {input_question} \n Answer: {input_answer} \n Generate one paragraph summary"
    result_summary = agent1.create(
        messages=[intermediate_result],
        context=None,
        experiment_id=experiment_id,
    )
    result = (
        "Here is the summary:\n"
        + result_summary
        + "\n\n From the above summary, generate few more questions to be asked to subject matter expert or reliability enginner or quality engineer."
    )
    question_response = agent2.create(
        messages=[result],
        context=None,
        experiment_id=experiment_id,
    )
    question_response_1 = agent2.extract_questions(question_response)
    return (result_summary, question_response, question_response_1)


def remove_duplicates(paragraphs):
    unique_paragraphs = []
    for paragraph in paragraphs:
        if not any(
            difflib.SequenceMatcher(None, paragraph, p).ratio() > 0.9
            for p in unique_paragraphs
        ):
            unique_paragraphs.append(paragraph)
    return unique_paragraphs


def post_process_duplicate(text):
    paragraphs = text.split("\n\n")
    unique_paragraphs = remove_duplicates(paragraphs)
    cleaned_output = "\n\n".join(unique_paragraphs)
    return cleaned_output


def clean_user_assistant(text):
    lines = text.split("\n")
    cleaned_text = "\n".join(
        line
        for line in lines
        if not (line.startswith("User:") or (line == "Assistant:"))
    )
    return cleaned_text


class RecipeAgent:
    LLMsets = [
        "ibm/granite-3-3-8b-instruct",
        "mistralai/mistral-large",
        "mistralai/mistral-medium-2505",
        "meta-llama/llama-3-3-70b-instruct",
    ]
    load_dotenv(".env")
    model_id = 3
    # configuration
    DEFAULT_CONFIG = {
        "model": LLMsets[model_id],
        "params": {
            "decoding_method": "greedy",
            "max_tokens": 2000,  # 1500,
            "stop": ["(TOKENSTOP)", "User:", "USER:", "Assistant:", "ASSISTANT:"],
            # "stream": True,
            # "return_options": {"input_text": False, "input_tokens": True},
            # "moderations": ModerationParameters(
            #     hap=ModerationHAP(input=True, output=False, threshold=0.01)
            # ),
        },
    }

    # Pump (Centrifugal)
    # Electrical submersible pump
    # Operator maintaining, design technicial, reliability engineer, technician : RCM Session
    # Note: Please do not use a conversational approach to ask questions and gather information.
    # I'll ask follow-up questions based on the response I receive to ensure that I have a clear
    # understanding of the problem and the data.

    SMESystemPrompt = """
    You act as a subject matter expert who is expert in failure modes and effect analysis (FMEA) of asset or system 
    reliability. 
    
    Here is FMEA Procedure:

    An FMEA is a systematic method of identifying and preventing product and process reliability 
    problems before they occur. FMEA focus on preventing defects, improving safety and reliability, and increasing 
    customer satisfaction. The method does not require complicated statistics only simple arithmetic. 
    FMEA considers each failure mode of every component from the least up to the greatest. Your task is to provide 
    accurate information about asset's component, subcomponent, assembly, failure location, failure mode, failure cause, failure influence so 
    immediate intervention in terms of maintenance or operations can be taken to extend 
    the life of the asset and or system. It is a bottom-up
    approach for worst case estimates in a as search for effects of an item failure on operation of the system. 
    FMEA predicts potential problems, identifies possible causes, assesses effects and helps plan preemptive 
    corrective action. 
    
    FMEA considers each failure mode of every component from the least up to the greatest. FMEA predicts 
    potential problems and identifies possible causes. By using collected information about the equipment, 
    failure modes are identified, and the appropriate tasks are selected to identify or prevent these 
    failure modes as early as possible. 
    
    User will provide a asset class or ask FMEA related question and you will provide descriptive FMEA 
    information by following above FMEA Procedure. 
    """

    QESystemPrompt = """
    You are a quality engineer responsible for generating FMEA documentation for a given asset class. 
    Your expertise in quality standards and regulations, including ISO 9001, IATF 16949, and 
    FMEA guidelines, will be utilized to ensure that the FMEA documentation is accurate, complete, and compliant 
    with quality standards and regulations. You will provide information to identify potential 
    failure locations, failure modes, their causes, and effects.
    """

    RESystemPrompt = """You act as a reliability engineer who provide information on 
    failure locations, failure modes, their causes, and effects and other failure related 
    information for the given asset class, with a focus on generating FMEA documentation. 
    You will analyzing the asset class to identify potential 
    failure modes and their impact on reliability.  
    """

    SessionSystemPrompt = """You act as an expert for generating failure mode
        and effect analysis (FMEA). Given an Asset Class and Asset Description information, 
        you will prepare a series of questions to be asked to subject matter expert. 
        Typical questions should focus on the important asset's component, subcomponent, 
        assembly, failure location, failure mode, failure cause and failure effect. Please do not use a conversational 
        approach to ask questions and gather information.
    """

    InfoSummaryPromt = """
Prepare a human-readable summary in a well-structured paragraph, eliminating any 
 special characters such as new lines and tabs. Focus on capturing the main component, subcomponent, 
 assembly, failure location, failure mode, failure cause, failure code. Ensure the summary provides a coherent narrative. 
 If user gives list of questions, then summary should be written based on questions content for a given asset class. 
"""
    # If user gives list of questions, then summary should be written based on questions content.
    # Generate one paragraph summary.
    # Generate the background document to answer the given question.

    QuestionGenerator = """
Pretend you are a question generation system. I will give you a list of questions or a pair of question and answer 
extracted from the conversation between two users where question is asked by facilitator and 
subject matter expert has provided corresponding answer. Based on the conversation, you reply me with additional set of 
questions that facilitator can ask to subject matter expert. The newly generated questions must align with original set of questions. 
you should avoid generating duplicate questions. you should also avoid questions for which potential answer can be similar.
 Please do not use a conversational approach to ask questions and gather information.
"""

    QuestionClassifier = """
You are a helpful, respectful, and honest assistant. You will be introduced to several 
persona such as subject matter experts, quality engineers or reliability engineers, etc. 
User will provide a question and you will select a persona who can answer the given question. Your selection is based 
on the persona's field experience and scientific knowledge. Sometime questions can be answered by 
multiple personas. Here is the two personas along with their skill description.  

Persona: FMEA Expert
Skill: Leading the FMEA session and ensuring that the process is followed correctly

Persona: Subject Matter Expert
Skill: Identifying potential failure modes, their effects, and their likelihood of occurrence.

Persona: Quality Engineers
Skill: Ensures that the FMEA is conducted in accordance with quality standards and regulations

Persona: Reliability Engineers
Skill: Provide information on failure rates, mean time between failures (MTBF), and other reliability metrics
"""

    QuestionClassifierPropmt = """
Assuming user is seeking information about who can answer the following question. There is a possibility
that more than one persona can provide different level of information. Please generate only personas.

Question: How do you ensure that the wind turbine gearbox is properly aligned and balanced? What are the consequences of misalignment or imbalance, and how do you correct these issues?

Question: Can you discuss the role of condition monitoring in predicting and preventing wind turbine gearbox failures? What are the different types of condition monitoring techniques, and how do they help identify potential failures?

Question: Can you provide examples of common mistakes or oversights that can lead to wind turbine gearbox failures? How can these mistakes be avoided, and what are the consequences of not addressing them?

Question: Are there any additional data sources or information that can be leveraged to improve the accuracy of the anomaly model, such as historical data or expert knowledge? This will help me identify potential sources of additional information that can be used to improve the model.
"""

    AssetDescriptionExtractor = """
You are a helpful, respectful, and honest assistant. You will be provided a one line description that 
include industrial asset name and may include some configuration such as component name or configuration. You need  
to identify the device name which represent an industrial asset from the given description and then 
generate a device type and short device description. Provide answer in python json string with following keys : 
asset_class, asset_category, device_name, device_type and device_description. 

If you don't know the answer to a question, please don't share false information. Your answers should not 
include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that 
your responses are socially unbiased and positive in nature.
"""

    QAClassifierSystemPrompt = """
You are a helpful, respectful, and honest assistant. You will be introduced to several 
persona such as fmea expert, subject matter experts, quality engineers or reliability engineers, etc. 
User will provide a question and you will select a persona who can answer the given question. 
Your selection is based on the persona's field experience and scientific knowledge. 
Sometime questions can be answered by multiple personas. Here is the four personas along with 
their skill description.  

Persona: FMEA Expert
Skill: Leading the FMEA session and ensuring that the process is followed correctly

Persona: Subject Matter Expert
Skill: Identifying potential failure modes, their effects, and their likelihood of occurrence

Persona: Quality Engineers
Skill: Ensures that the FMEA is conducted in accordance with quality standards and regulations

Persona: Reliability Engineers
Skill: Provide information on failure rates, mean time between failures (MTBF), and other reliability metrics

"""

    QAClassifierPrompt = """
Assuming user is seeking guidance about who can answer the given question. There is a possibility
that more than one personas can provide different level of information. Your answer should include 
all personas who can be the best persona to consult for the question. You will use the example provided 
in a form of Internal thought to find the answer for all questions.

Question: What are the most common failure modes for wind turbine gearboxes? This will help me identify the failure modes that the anomaly model should be able to detect. 
Please use (Internal thought).

(Internal thought): first we find out the list of candidate personas mentioned in System Prompt. 

We found four personas listed in system prompt: [Subject Matter Expert, FMEA Facilitator, Quality Engineers, Reliability Engineers]. 

First, let us evaluate first persona (Subject Matter Expert). The Subject Matter Expert has knowledge of the domain and can provide
 information about the common failure modes for wind turbine gearboxes. This question is best suited for a Subject Matter Expert.
 The sentiment for Subject Matter Expert is positive.

Second, we evaluate second persona (FMEA Facilitator). This question is primarily related to the domain knowledge of
 wind turbine gearboxes and their failure modes. Therefore, FMEA Facilitator is not the best persona to consult for this question. 
 The sentiment for FMEA Facilitator is negative. 

Next, we evaluate the third persona (Quality Engineers). Quality Engineers mainly focus on manufacturing quality and process issues. Their link to operational gearbox failure modes is limited. The sentiment for Quality Engineers is negative.

Finally, we evaluate the fourth persona (Reliability Engineers). Reliability Engineers have expertise in equipment performance, failure mechanisms, and predictive maintenance, making them well-suited for this question. The sentiment for Reliability Engineers is positive.

Overall, Subject Matter Expert and Reliability Engineers both have positive sentiment.

Answer: The final answer is Subject Matter Expert and Reliability Engineers. (TOKENSTOP)

Question: What quality control checks are performed during the manufacturing of wind turbine gearboxes? Please use (Internal thought).

(Internal thought): first we find out the list of candidate personas mentioned in System Prompt. 

We found four personas listed in system prompt: [Subject Matter Expert, FMEA Facilitator, Quality Engineers, Reliability Engineers]. 

Subject Matter Expert has domain knowledge but is more focused on operational performance and failure modes rather than detailed manufacturing QC checks. The sentiment for Subject Matter Expert is negative.

FMEA Facilitator: Works on risk analysis and failure mode mapping, but not primarily involved in routine QC processes. The sentiment for FMEA Facilitator is negative

Quality Engineers: Specialize in manufacturing quality control, process compliance, and defect prevention. This question directly aligns with their responsibilities → positive. The sentiment for Quality Engineers is positive

Reliability Engineers: Focus on asset performance and reliability during operation, not on manufacturing. The sentiment for Reliability Engineers is negative.

Overall, Quality Engineers has positive sentiment.

Answer: The final answer is Quality Engineers. (TOKENSTOP)
"""

    # The asset description is "Valve - Hydraulic Operated - Isolation - Piston Type".
    def __init__(
        self,
        name: str,
        custom_config=None,
        model_id=3,
    ):
        self.name = name
        self.model_id = model_id
        self.DEFAULT_CONFIG["model"] = self.LLMsets[model_id]
        self.genai_config = dict(self.DEFAULT_CONFIG)
        if custom_config:
            self.genai_config.update(custom_config)

        self._genai_messages = defaultdict(list)

        # agent
        self.DSAgent = None

        # agent 1
        self.SFAgent = GenAIChatClient(
            name="FR",
            description="Facilitator",
            skill="Leading the FMEA session and ensuring that the process is followed correctly",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            system_message=self.SessionSystemPrompt,
            post_process_text=True,
            # question_message=None,
        )

        # agent 2
        self.SMEAgent = GenAIChatClient(
            name="SME",
            description="Subject Matter Expert",
            skill="Identifying potential failure modes, their effects, and their likelihood of occurrence",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            system_message=self.SMESystemPrompt,
            post_process_text=True,
            # question_message=None,
        )

        # agent 3
        self.SummarizeAgent = GenAIChatClient(
            name="Summarizer",
            description="Answer Summarizer",
            skill="generate summary of provided document, document summarization task",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            system_message=self.InfoSummaryPromt,
            post_process_text=True,
        )

        # agent 4
        tmp_genai_config = dict(self.DEFAULT_CONFIG)
        tmp_genai_config["params"]["max_tokens"] = 500
        self.QuestionGeneratorAgent = GenAIChatClient(
            name="QA",
            description="Question Answer Generator",
            skill="generate new set of questions from input documents",
            model=tmp_genai_config["model"],
            params=tmp_genai_config["params"],
            system_message=self.QuestionGenerator,
            post_process_text=True,
        )

        # agent 5
        self.QuestionClassifierAgent = GenAIInstructClient(
            name="QClassifier",
            description="Find right persona for a given question",
            skill="use LLM to classifier a question into persona",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            system_message=self.QAClassifierSystemPrompt,
            question_message=self.QAClassifierPrompt,
        )

        # two more: Reliability and Quality Controller
        self.QEAgent = GenAIChatClient(
            name="QE",
            description="Quality Engineer",
            skill="Ensures that the FMEA is conducted in accordance with quality standards and regulations",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            system_message=self.QESystemPrompt,
            post_process_text=True,
        )

        self.REAgent = GenAIChatClient(
            name="RE",
            description="Reliability Engineer",
            skill="Provide information on failure rates, mean time between failures (MTBF), and other reliability metrics",
            model=self.genai_config["model"],
            params=self.genai_config["params"],
            system_message=self.RESystemPrompt,
            post_process_text=True,
        )

        # messages - storage
        self.question_placeholder_ = (
            []
        )  # store all the questions generated in current round
        self.question_track_round_info_placeholder_ = []  # store the question round

        self.genai_questions_for_sme = []  # store all the questions to be asked to SME
        self.genai_responses_from_sme = []  # store all the response from SME
        self.genai_track_round_info_sme = []  # store all the response from SME

        self.genai_questions_for_ds = []  # store all the questions to be asked to DS
        self.genai_responses_from_ds = []  # store all the response from DS
        self.genai_track_round_info_ds = []  # store all the response from DS

        self.genai_questions_for_sf = []  # store all the questions to be asked to DS
        self.genai_responses_from_sf = []  # store all the response from DS
        self.genai_track_round_info_sf = []  # store all the response from DS

        self.genai_questions_for_re = []  # store all the questions to be asked to DS
        self.genai_responses_from_re = []  # store all the response from DS
        self.genai_track_round_info_re = []  # store all the response from DS

        self.genai_questions_for_qe = []  # store all the questions to be asked to DS
        self.genai_responses_from_qe = []  # store all the response from DS
        self.genai_track_round_info_qe = []  # store all the response from DS

        self.genai_questions_outof_scope = []  # this is out of scope question
        self.question_generation_track = []  # track number of questions being generated

        # sme counter
        self.total_processed_question_sme = 0
        self.total_processed_answer_sme = 0

        # ds counter
        self.total_processed_answer_ds = 0
        self.total_processed_question_ds = 0

        # sf counter
        self.total_processed_answer_sf = 0
        self.total_processed_question_sf = 0

        # re counter
        self.total_processed_answer_re = 0
        self.total_processed_question_re = 0

        # qe counter
        self.total_processed_answer_qe = 0
        self.total_processed_question_qe = 0

        # this is to print the all intermediate message for debug and imporovement
        self.testmode = 1
        ray.init(ignore_reinit_error=True, num_cpus=1, local_mode=True)

    def set_asset_class(self, asset_class):
        self.asset_class = asset_class

    def init_round(self, message, experiment_id, asset_description):
        """
        This is a round 1 which inform a basic initialization to the DS and SME agents.
        """

        """Zero shot"""
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- Round 1 ------------------------------.{Style.RESET_ALL}"
            )

        # adding this to update
        self.SFAgent.update_system_message(self.asset_class, self.asset_description_)
        # self.SMEAgent.update_system_message(self.asset_class, self.asset_description_)
        # self.QEAgent.update_system_message(self.asset_class, self.asset_description_)
        # self.REAgent.update_system_message(self.asset_class, self.asset_description_)

        # inform SFAgent about the asset class
        # sfagent
        sf_response = self.SFAgent.create(
            messages=["Generate Questions."],
            context=None,
            experiment_id=experiment_id,
        )
        print(sf_response)

        # inform SME about the asset class
        sme_response = self.SMEAgent.create(
            messages=[message],
            context=None,
            experiment_id=experiment_id,
        )
        self.sme_response_ = sme_response

        # quality engineer
        qe_response = self.QEAgent.create(
            messages=[message],
            context=None,
            experiment_id=experiment_id,
        )
        self.qe_response_ = qe_response

        # reliability engineer
        re_response = self.REAgent.create(
            messages=[message],
            context=None,
            experiment_id=experiment_id,
        )
        self.re_response_ = re_response

        if self.testmode:
            print(f"{Style.BRIGHT}{Fore.BLUE} sf_response >>> {Style.RESET_ALL}")
            print(sf_response)
            print(f"{Style.BRIGHT}{Fore.BLUE} sme_response >>> {Style.RESET_ALL}")
            print(sme_response)
            print(f"{Style.BRIGHT}{Fore.BLUE} QE_response >>> {Style.RESET_ALL}")
            print(qe_response)
            print(f"{Style.BRIGHT}{Fore.BLUE} RE_response >>> {Style.RESET_ALL}")
            print(re_response)

        # extract initial set of questions prepared by DS
        sf_questions = self.SFAgent.extract_questions(sf_response)
        f_sf_question = filter_and_sort_questions(
            sf_questions,
            filter_threshold=0.98,
        )

        # put quection in placeholder
        self.question_placeholder_.extend(f_sf_question)

        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.BLUE} sf_questions : {len(sf_questions)} >>> {Style.RESET_ALL}"
            )
            print(sf_questions)
            print(
                f"{Style.BRIGHT}{Fore.BLUE} f_sf_question {len(f_sf_question)} >>> {Style.RESET_ALL}"
            )
            print(f_sf_question)

        """In Context Learning : Invoke Summarize Agent and then Get summary"""
        """Extending to multiple persona ..."""
        self.track_context_summary_ = []
        for agent_name, agent_full_name, agent_response in [
            ("sme", "subject matter expert", sme_response),
            ("re", "reliability engineer", re_response),
            ("qe", "quality engineer", qe_response),
        ]:
            fs_summary = self.SummarizeAgent.create(
                messages=[agent_response + "\n Generate one paragraph summary."],
                context=None,
                experiment_id=experiment_id,
            )
            self.context_documents_ = fs_summary
            self.track_context_summary_.append(
                (agent_name, agent_full_name, agent_response, fs_summary)
            )

            if self.testmode:
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} fs_summary : {len(fs_summary)} >>> {Style.RESET_ALL}"
                )
                print(fs_summary)

            """ Now ask DS """
            addon = f"Here is the summary: \n {fs_summary} \n\n From the above given summary, generate few more questions to be asked to {agent_full_name}."
            sf_response_1 = self.SFAgent.create(
                messages=[addon],
                context=None,
                experiment_id=experiment_id,
            )

            sf_questions_1 = self.SFAgent.extract_questions(sf_response_1)

            # finalize the question preparations
            f_sf_question_1 = filter_questions_using_reference(
                self.question_placeholder_, sf_questions_1, filter_threshold=0.98
            )
            f_sf_question_2 = filter_and_sort_questions(
                f_sf_question_1, filter_threshold=0.98
            )
            self.question_placeholder_.extend(f_sf_question_2)

            # apply filtering ()
            self.question_placeholder_ = filter_questions_using_TTR(
                self.question_placeholder_
            )
            self.question_placeholder_ = filter_questions_using_Flan(
                self.question_placeholder_
            )

            if self.testmode:
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} Agent Name :  {agent_name} >>> {Style.RESET_ALL}"
                )
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} sf_response_1 : {len(sf_response_1)} >>> {Style.RESET_ALL}"
                )
                print(sf_response_1)
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} sf_questions_1 : {len(sf_questions_1)} >>> {Style.RESET_ALL}"
                )
                print(sf_questions_1)
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} f_sf_question_1 : {len(f_sf_question_1)} >>> {Style.RESET_ALL}"
                )
                print(f_sf_question_1)
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} f_sf_question_2 : {len(f_sf_question_2)} >>> {Style.RESET_ALL}"
                )
                print(f_sf_question_2)

        self.context_questions_seeds_total_ = len(self.question_placeholder_)

        # now we generate context document
        result = "\n".join(
            [f"{i+1}. {item}" for i, item in enumerate(self.question_placeholder_)]
        )
        result = (
            "Here is the list of questions: \n"
            + result
            + "\n Generate one paragraph summary."
        )
        # result += "\n Generate one paragraph summary."
        ds_context_summary = self.SummarizeAgent.create(
            messages=[result],
            context=None,
            experiment_id=experiment_id,
        )
        self.question_context_documents_ = ds_context_summary
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.BLUE} Question Context :  {len(self.question_placeholder_)} >>> {Style.RESET_ALL}"
            )
            print(
                f"{Style.BRIGHT}{Fore.BLUE} Question context : {len(ds_context_summary)} >>> {Style.RESET_ALL}"
            )
            print(
                f"{Style.BRIGHT}{Fore.BLUE} Question context Summary : {ds_context_summary} >>> {Style.RESET_ALL}"
            )

    def answer_generation(self, experiment_id):
        """
        This part is to generate a response of user defined question
        We have two persons with whom we need to generate answer
        We have a pointer from where onward we should process
        """
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- Answer Generation Start ------------------------------.{Style.RESET_ALL}"
            )

        """This is a round 2 - Where SME answer"""
        refs = []
        for item in self.genai_questions_for_sme[self.total_processed_question_sme :]:
            refs.append(
                generate_response.remote(
                    self.SMEAgent,
                    [self.context_prompt_ + ". " + item],
                    experiment_id,
                )
            )
        sme_responses = ray.get(refs)
        self.genai_responses_from_sme.extend(sme_responses)
        if self.testmode:
            for item_index, item in enumerate(
                self.genai_questions_for_sme[self.total_processed_question_sme :]
            ):
                print(f"{Style.BRIGHT}{Fore.BLUE} Question : >>> {Style.RESET_ALL}")
                print(item)
                print(f"{Style.BRIGHT}{Fore.BLUE} Answer : >>> {Style.RESET_ALL}")
                print(sme_responses[item_index])

        """This is a round 2 - Where FS answer """
        refs = []
        for item in self.genai_questions_for_sf[self.total_processed_answer_sf :]:
            tmp_ds_question = "Provide a background document to answer the given question. \n\n Question: "
            refs.append(
                generate_response.remote(
                    self.SFAgent,
                    [tmp_ds_question + " " + self.context_prompt_ + ". " + item],
                    experiment_id,
                )
            )
        fs_responses = ray.get(refs)
        if self.testmode:
            for item_index, item in enumerate(
                self.genai_questions_for_sf[self.total_processed_answer_sf :]
            ):
                print(f"{Style.BRIGHT}{Fore.BLUE} Question : >>> {Style.RESET_ALL}")
                print(item)
                print(f"{Style.BRIGHT}{Fore.BLUE} Answer : >>> {Style.RESET_ALL}")
                print(fs_responses[item_index])
        self.genai_responses_from_sf.extend(fs_responses)

        # to add the code here ...
        """This is a round 2 - Where QE answer"""
        refs = []
        for item in self.genai_questions_for_qe[self.total_processed_question_qe :]:
            refs.append(
                generate_response.remote(
                    self.QEAgent,
                    [self.context_prompt_ + ". " + item],
                    experiment_id,
                )
            )
        qe_responses = ray.get(refs)
        self.genai_responses_from_qe.extend(qe_responses)
        if self.testmode:
            for item_index, item in enumerate(
                self.genai_questions_for_qe[self.total_processed_question_qe :]
            ):
                print(f"{Style.BRIGHT}{Fore.BLUE} Question : >>> {Style.RESET_ALL}")
                print(item)
                print(f"{Style.BRIGHT}{Fore.BLUE} Answer : >>> {Style.RESET_ALL}")
                print(qe_responses[item_index])

        """This is a round 2 - Where RE answer"""
        refs = []
        for item in self.genai_questions_for_re[self.total_processed_question_re :]:
            refs.append(
                generate_response.remote(
                    self.REAgent,
                    [self.context_prompt_ + ". " + item],
                    experiment_id,
                )
            )
        re_responses = ray.get(refs)
        self.genai_responses_from_re.extend(re_responses)
        if self.testmode:
            for item_index, item in enumerate(
                self.genai_questions_for_re[self.total_processed_question_re :]
            ):
                print(f"{Style.BRIGHT}{Fore.BLUE} Question : >>> {Style.RESET_ALL}")
                print(item)
                print(f"{Style.BRIGHT}{Fore.BLUE} Answer : >>> {Style.RESET_ALL}")
                print(re_responses[item_index])

        # to add the code here ...

        self.total_processed_question_sf = len(self.genai_responses_from_sf)
        self.total_processed_question_sme = len(self.genai_responses_from_sme)
        self.total_processed_question_qe = len(self.genai_responses_from_qe)
        self.total_processed_question_re = len(self.genai_responses_from_re)

        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- Answer Generation End ------------------------------.{Style.RESET_ALL}"
            )

    def _generate_questions(
        self, input_question_sets, input_answer_sets, index_to_be_used, experiment_id
    ):
        """_summary_

        :param input_question_sets: _description_
        :type input_question_sets: _type_
        :param index_to_be_used: _description_
        :type index_to_be_used: _type_
        """
        tmp_DSets = []

        # Approach 1
        # all questions in (random order) and let context to cut it
        # at some point it will be over the context and then system will remove or raise flag
        if len(input_question_sets) < 30:
            randomized_questions = random.sample(
                input_question_sets, len(input_question_sets)
            )
            result = "\n".join(
                [f"{i+1}. {item}" for i, item in enumerate(randomized_questions)]
            )
            result = (
                "Here is a list of questions: \n"
                + result
                + "\n\n From the above list of questions, generate few more questions to be asked to subject matter expert or reliability enginner or quality engineer."
            )
            question_response = self.QuestionGeneratorAgent.create(
                messages=[result],
                context=None,
                experiment_id=experiment_id,
            )
            question_response_1 = self.QuestionGeneratorAgent.extract_questions(
                question_response
            )
            tmp_DSets.extend(question_response_1)
            if self.testmode:
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_input :  >>> {Style.RESET_ALL}"
                )
                print(result)
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_response >>> {Style.RESET_ALL}"
                )
                print(question_response)
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_response_1 : {len(question_response_1)} >>> {Style.RESET_ALL}"
                )
                print(question_response_1)

        """
        # approach 2
        # most recent first
        result = "\n".join(["1. " + input_question_sets[-1]])
        question_response = self.QuestionGeneratorAgent.create(
            messages=[{"content": result, "role": "user", "type": "qq"}],
            context=None,
            experiment_id=experiment_id,
        )
        question_response_2 = self.QuestionGeneratorAgent.extract_questions(
            question_response
        )
        tmp_DSets.extend(question_response_2)
        if self.testmode:
            print(f"{Style.BRIGHT}{Fore.BLUE} question_input :  >>> {Style.RESET_ALL}")
            print(result)
            print(f"{Style.BRIGHT}{Fore.BLUE} question_response >>> {Style.RESET_ALL}")
            print(question_response)
            print(
                f"{Style.BRIGHT}{Fore.BLUE} question_response : {len(question_response_1)} >>> {Style.RESET_ALL}"
            )
            print(question_response_2)
        """

        # approach 3 - 10 most recent questions
        # most recent first
        for _ in range(len(input_question_sets) // 100 + 1):
            selected_elements = random.sample(
                input_question_sets, min(len(input_question_sets), 30)
            )
            result = "\n".join(
                [f"{i+1}. {item}" for i, item in enumerate(selected_elements)]
            )
            result = (
                "Here is a list of questions: \n"
                + result
                + "\n\n From the above list of questions, generate few more questions to be asked to subject matter expert or reliability enginner or quality engineer."
            )
            question_response = self.QuestionGeneratorAgent.create(
                messages=[result],
                context=None,
                experiment_id=experiment_id,
            )
            question_response_3 = self.QuestionGeneratorAgent.extract_questions(
                question_response
            )
            tmp_DSets.extend(question_response_3)
            if self.testmode:
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_input :  >>> {Style.RESET_ALL}"
                )
                print(result)
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_response >>> {Style.RESET_ALL}"
                )
                print(question_response)
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_response : {len(question_response_3)} >>> {Style.RESET_ALL}"
                )
                print(question_response_3)

        # approach 4. Q1, A1 --> Q2
        # purely using question-answer pair
        tmp_tmp_DSets = []
        tmp_tmp_RSet = []
        tmp_tmp_QSet = []

        refs = []
        for qid in range(index_to_be_used, len(input_question_sets)):
            refs.append(
                generate_chain_response.remote(
                    self.SummarizeAgent,
                    self.QuestionGeneratorAgent,
                    input_question_sets[qid],
                    input_answer_sets[qid],
                    experiment_id,
                )
            )
        ray_responses = ray.get(refs)
        for qid in range(len(ray_responses)):
            tmp_tmp_RSet.append(ray_responses[qid][0])
            tmp_tmp_QSet.append(ray_responses[qid][1])
            tmp_tmp_DSets.append(ray_responses[qid][2])

        # generating a global string
        for qid in range(len(tmp_tmp_DSets)):
            tmp_DSets.extend(tmp_tmp_DSets[qid])

        if self.testmode:
            for qid in range(index_to_be_used, len(input_question_sets)):
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_input_1 :  >>> {Style.RESET_ALL}"
                )
                print(
                    f"Question: {input_question_sets[qid]} \n Answer: {input_answer_sets[qid]}"
                )
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_summary  :  >>> {Style.RESET_ALL}"
                )
                print(tmp_tmp_RSet[qid - index_to_be_used])
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_input : {len(tmp_tmp_QSet[qid - index_to_be_used])} >>> {Style.RESET_ALL}"
                )
                print(tmp_tmp_QSet[qid - index_to_be_used])
                print(
                    f"{Style.BRIGHT}{Fore.BLUE} question_response : {len(tmp_tmp_DSets[qid - index_to_be_used])} >>> {Style.RESET_ALL}"
                )
                print(tmp_tmp_DSets[qid - index_to_be_used])

        return tmp_DSets

    def question_generation(self, experiment_id):
        """
        This process use the SME and DS Response to generate the
        This is a question generation
        """
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- Question Generation Start ------------------------------.{Style.RESET_ALL}"
            )

        tmp_DSets = []
        # purelly using questions

        # approch 1. Q1, Q2, Q3, ..... , Q20 ---> Q21, ...., Q30 (Question Prediction)
        # we use the questions that were designed to ask SME

        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- SME ------------------------------.{Style.RESET_ALL}"
            )
            print(len(self.genai_questions_for_sme), len(self.genai_responses_from_sme))

        # Generate question for SME
        set1 = self._generate_questions(
            self.genai_questions_for_sme,
            self.genai_responses_from_sme,
            self.total_processed_answer_sme,
            experiment_id,
        )
        self.total_processed_answer_sme = len(self.genai_responses_from_sme)

        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- SF ------------------------------.{Style.RESET_ALL}"
            )
            print(len(self.genai_questions_for_sf), len(self.genai_responses_from_sf))

        # Generate question for DS
        set2 = self._generate_questions(
            self.genai_questions_for_sf,
            self.genai_responses_from_sf,
            self.total_processed_answer_sf,
            experiment_id,
        )
        self.total_processed_answer_sf = len(self.genai_responses_from_sf)

        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- QE ------------------------------.{Style.RESET_ALL}"
            )
            print(len(self.genai_questions_for_qe), len(self.genai_responses_from_qe))

        # Generate question for QE
        set3 = self._generate_questions(
            self.genai_questions_for_qe,
            self.genai_responses_from_qe,
            self.total_processed_answer_qe,
            experiment_id,
        )
        self.total_processed_answer_qe = len(self.genai_responses_from_qe)

        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- RE ------------------------------.{Style.RESET_ALL}"
            )
            print(len(self.genai_questions_for_re), len(self.genai_responses_from_re))

        # Generate question for DS
        set4 = self._generate_questions(
            self.genai_questions_for_re,
            self.genai_responses_from_re,
            self.total_processed_answer_re,
            experiment_id,
        )
        self.total_processed_answer_re = len(self.genai_responses_from_re)

        # adding the results into the common set
        tmp_DSets.extend(set1)
        tmp_DSets.extend(set2)
        tmp_DSets.extend(set3)
        tmp_DSets.extend(set4)

        # post process the questions and then
        # remove duplicate question with high threshold
        unique_list = list(
            OrderedDict.fromkeys(
                self.genai_questions_for_sme
                + self.genai_questions_for_sf
                + self.genai_questions_for_re
                + self.genai_questions_for_qe
            )
        )
        f_tmp_DSets_1 = filter_questions_using_reference(
            unique_list,
            tmp_DSets,
            filter_threshold=0.98,
        )
        # we prefer the longer questions
        f_tmp_DSets_2 = filter_and_sort_questions(f_tmp_DSets_1, filter_threshold=0.98)
        self.question_placeholder_.extend(f_tmp_DSets_2)

        # ttr replacement
        self.question_placeholder_ = filter_questions_using_TTR(
            self.question_placeholder_
        )
        # flan replacement
        self.question_placeholder_ = filter_questions_using_Flan(
            self.question_placeholder_
        )

        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.BLUE} Question so far in Bank : {len(unique_list)} >>> {Style.RESET_ALL}"
            )
            print(
                f"{Style.BRIGHT}{Fore.BLUE} total Questions generated : {len(tmp_DSets)} >>> {Style.RESET_ALL}"
            )
            print(
                f"{Style.BRIGHT}{Fore.BLUE} f_tmp_DSets_1 : {len(f_tmp_DSets_1)} >>> {Style.RESET_ALL}"
            )
            print(
                f"{Style.BRIGHT}{Fore.BLUE} f_tmp_DSets_2 : {len(f_tmp_DSets_2)} >>> {Style.RESET_ALL}"
            )
            print(f"Total New Questions : {len(self.question_placeholder_)}")
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- Question Generation End ------------------------------.{Style.RESET_ALL}"
            )

    def question_assignment(self, experiment_id, total_round=0):
        """_summary_ This code will iterate over all the questions in question_placeholder_

        :param experiment_id: _description_
        :type experiment_id: _type_
        """
        if self.testmode:
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- Question Asignment Round ------------------------------.{Style.RESET_ALL}"
            )

        total_ds_q = 0
        total_sme_q = 0
        total_sf_q = 0
        total_re_q = 0
        total_qe_q = 0

        total_outside_q = 0
        total_overlap_q = 0

        llm_answers = []

        refs = []
        for question in self.question_placeholder_:
            refs.append(
                generate_response.remote(
                    self.QuestionClassifierAgent, question, experiment_id
                )
            )
        llm_answers = ray.get(refs)

        # now we have answer and we can assign it to the right place
        for llm_index, llm_answer in enumerate(llm_answers):
            # extract the text
            sindex = llm_answer.rfind("Answer:")
            eindex = llm_answer.rfind("(TOKENSTOP")
            answer_text = llm_answer[sindex + 7 : eindex].lower()  # 7 = len('Answer:')

            other_q1 = False
            if "subject matter expert" in answer_text:
                self.genai_questions_for_sme.append(
                    self.question_placeholder_[llm_index]
                )
                self.genai_track_round_info_sme.append(total_round)
                total_sme_q = total_sme_q + 1
            else:
                other_q1 = True

            other_q2 = False
            if "fmea expert" in answer_text or "fmea facilitator" in answer_text:
                self.genai_questions_for_sf.append(
                    self.question_placeholder_[llm_index]
                )
                self.genai_track_round_info_sf.append(total_round)
                total_sf_q = total_sf_q + 1
            else:
                other_q2 = True

            other_q3 = False
            if "reliability engineer" in answer_text:
                self.genai_questions_for_re.append(
                    self.question_placeholder_[llm_index]
                )
                self.genai_track_round_info_re.append(total_round)
                total_re_q = total_re_q + 1
            else:
                other_q3 = True

            other_q4 = False
            if "quality engineer" in answer_text:
                self.genai_questions_for_qe.append(
                    self.question_placeholder_[llm_index]
                )
                self.genai_track_round_info_qe.append(total_round)
                total_qe_q = total_qe_q + 1
            else:
                other_q4 = True

            if other_q1 and other_q2 and other_q3 and other_q4:
                total_outside_q = total_outside_q + 1
                self.genai_questions_outof_scope.append(
                    self.question_placeholder_[llm_index]
                )

            if not other_q1 and not other_q2 and not other_q3 and not other_q4:
                total_overlap_q = total_overlap_q + 1

            if self.testmode:
                print(f"Question: {self.question_placeholder_[llm_index]}")
                print(f" >>> Answer: {answer_text}")

        if self.testmode:
            print(
                f"Questions for SME : {total_sme_q}, FMEA Session : {total_sf_q}, Reliability Engineer : {total_re_q}, Quality Engineer : {total_qe_q}, Other Question : {total_outside_q}, Overlap Question : {total_overlap_q}"
            )
            print(
                f"{Style.BRIGHT}{Fore.GREEN}--------------------- Question Asignment End Round ------------------------------.{Style.RESET_ALL}"
            )

        self.question_generation_track.append(
            {
                "round": total_round,
                "total_sme_q": total_sme_q,
                "total_fm_q": total_ds_q,
                "total_re_q": total_re_q,
                "total_qe_q": total_qe_q,
                "total_outside_q": total_outside_q,
                "total_overlap_q": total_overlap_q,
                "total_questions": (len(self.question_placeholder_)),
            }
        )

    def print_token_usage(self):
        self.DSAgent.print_token_usage()
        self.SMEAgent.print_token_usage()

    def init_chat(self, round=2, add_description=False):
        """_summary_"""

        # this is a context prompt (agenda)
        self.context_prompt_ = "The industrial asset class is " + self.asset_class

        # this is a common routine to get the detail information about given asset class
        tmp_asset_desc = get_description(
            iteration=2, asset_class=self.asset_class, model_id=self.model_id
        )
        self.asset_description_ = tmp_asset_desc

        if add_description:
            self.context_prompt_ += f"{self.context_prompt_} \n\n Asset Description : \n {self.asset_description_} \n\n"

        # setting the MLFLow experiments
        experiment_name = "MyExperiment_" + str(uuid.uuid4())
        experiment_id = mlflow.create_experiment(experiment_name)

        print(f">>> Message: {self.context_prompt_}")
        print(f">>> Experiment id: {experiment_id}")
        print(f">>> Experiment name: {experiment_name}")
        print(f">>> Asset Class : {self.asset_class}")
        print(f">>> Asset Description : {self.asset_description_}")

        # start recording
        with mlflow.start_run(experiment_id=experiment_id):
            # Initial round (DSE and SME get ready for their meeting, and they do some background work)
            self.init_round(
                message=self.context_prompt_,
                experiment_id=experiment_id,
                asset_description=self.asset_description_,
            )

            if self.testmode:
                print(
                    f"{Style.BRIGHT}{Fore.MAGENTA} CP: <<< {self.context_prompt_} >>> {Style.RESET_ALL}"
                )
                for (
                    agent_name,
                    agent_full_name,
                    agent_response,
                    fs_summary,
                ) in self.track_context_summary_:
                    print(
                        f"{Style.BRIGHT}{Fore.MAGENTA} <<< {agent_name}-{agent_full_name} : >>> {Style.RESET_ALL}"
                    )
                    print(
                        f"{Style.BRIGHT}{Fore.BLUE} <<< {fs_summary} >>> {Style.RESET_ALL}"
                    )
                print(
                    f"{Style.BRIGHT}{Fore.CYAN} SeedQuestions: <<< {self.context_questions_seeds_total_} >>> {Style.RESET_ALL}"
                )
                print(
                    f"{Style.BRIGHT}{Fore.GREEN} CPDSDOC: <<< {self.question_context_documents_} >>> {Style.RESET_ALL}"
                )
                print(
                    f"{Style.BRIGHT}{Fore.LIGHTRED_EX} Questions: <<< {self.question_placeholder_} >>> {Style.RESET_ALL}"
                )

            # if number of questions are Zero, do some post analysis.... else move forward
            if len(self.question_placeholder_) > 0:
                # Now use initial seed questions for second round
                total_round = 0
                while total_round < round:
                    # question assignment: questions from placeholder will be assigned to SME/DS.
                    # after question_assignment round, reset the question placeholder
                    self.question_assignment(
                        experiment_id=experiment_id, total_round=total_round
                    )
                    self.question_placeholder_ = []
                    self.question_track_round_info_placeholder_ = []

                    # interact with SME and then talk to DS
                    self.answer_generation(experiment_id=experiment_id)

                    # now we have response from sme, so we can create few more questions
                    self.question_generation(experiment_id=experiment_id)

                    # condition to quit early
                    if len(self.question_placeholder_) == 0:
                        break

                    total_round = total_round + 1
                    self.question_track_round_info_placeholder_ = [
                        total_round for _ in range(len(self.question_placeholder_))
                    ]

        # this part is about flushing the information out for future work
        # temp variable
        model_initial = self.genai_config["model"].split("/")[1].split("-")[0]

        sme_tuples = list(
            zip(self.genai_questions_for_sme, self.genai_track_round_info_sme)
        )
        sf_tuples = list(
            zip(self.genai_questions_for_sf, self.genai_track_round_info_sf)
        )
        re_tuples = list(
            zip(self.genai_questions_for_re, self.genai_track_round_info_re)
        )
        qe_tuples = list(
            zip(self.genai_questions_for_qe, self.genai_track_round_info_qe)
        )
        placeholder_tuples = list(
            zip(self.question_placeholder_, self.question_track_round_info_placeholder_)
        )

        set1 = set(sme_tuples)
        set2 = set(sf_tuples)
        set3 = set(re_tuples)
        set4 = set(qe_tuples)
        set5 = set(placeholder_tuples)

        merged_set = set1.union(set2)
        merged_set1 = merged_set.union(set3)
        merged_set2 = merged_set1.union(set4)
        final_set = merged_set2.union(set5)
        final_list = list(final_set)
        if self.testmode:
            print(f"{Style.BRIGHT}{Fore.BLUE} SME Tuples >>> {Style.RESET_ALL}")
            print(sme_tuples)
            print(f"{Style.BRIGHT}{Fore.BLUE} FC Tuples >>> {Style.RESET_ALL}")
            print(sf_tuples)
            print(f"{Style.BRIGHT}{Fore.BLUE} RE Tuples >>> {Style.RESET_ALL}")
            print(re_tuples)
            print(f"{Style.BRIGHT}{Fore.BLUE} QE Tuples >>> {Style.RESET_ALL}")
            print(qe_tuples)
            print(
                f"{Style.BRIGHT}{Fore.BLUE} Total questions : {len(merged_set)} >>> {Style.RESET_ALL}"
            )

        # this is a final step
        if "with component boundry" in self.asset_class:
            tmp_asset_init = self.asset_class.split("with component boundry")[0]
        else:
            tmp_asset_init = self.asset_class
        if round > 0:
            df = pd.DataFrame(final_list, columns=["questions", "round"])
            try:
                df.to_csv(
                    f"genai_questions_bank_{tmp_asset_init.replace(' ', '')}_{model_initial}_{experiment_id}.csv",
                    index=False,
                )
            except:
                pass

            df = pd.DataFrame(
                {
                    "questions": self.genai_questions_for_sme,
                    "answers": self.genai_responses_from_sme,
                    "round": self.genai_track_round_info_sme,
                }
            )

            try:
                df.to_csv(
                    f"genai_questions_answer_sme_bank_{tmp_asset_init.replace(' ', '')}_{model_initial}_{experiment_id}.csv",
                    index=False,
                )
            except:
                pass

            df = pd.DataFrame(
                {
                    "questions": self.genai_questions_for_sf,
                    "answers": self.genai_responses_from_sf,
                    "round": self.genai_track_round_info_sf,
                }
            )
            try:
                df.to_csv(
                    f"genai_questions_answer_sf_bank_{tmp_asset_init.replace(' ', '')}_{model_initial}_{experiment_id}.csv",
                    index=False,
                )
            except:
                pass

            df = pd.DataFrame(
                {
                    "questions": self.genai_questions_for_re,
                    "answers": self.genai_responses_from_re,
                    "round": self.genai_track_round_info_re,
                }
            )

            try:
                df.to_csv(
                    f"genai_questions_answer_re_bank_{tmp_asset_init.replace(' ', '')}_{model_initial}_{experiment_id}.csv",
                    index=False,
                )
            except:
                pass

            df = pd.DataFrame(
                {
                    "questions": self.genai_questions_for_qe,
                    "answers": self.genai_responses_from_qe,
                    "round": self.genai_track_round_info_qe,
                }
            )
            try:
                df.to_csv(
                    f"genai_questions_answer_qe_bank_{tmp_asset_init.replace(' ', '')}_{model_initial}_{experiment_id}.csv",
                    index=False,
                )
            except:
                pass

            df = pd.DataFrame(self.question_generation_track)
            try:
                df.to_csv(
                    f"genai_questions_track_{tmp_asset_init.replace(' ', '')}_{model_initial}_{experiment_id}.csv",
                    index=False,
                )
            except:
                pass

        df = pd.DataFrame(
            {
                "context_prompt": [self.context_prompt_],
                "sf_context": [self.context_documents_],
                "question_context": [self.question_context_documents_],
                "sme_context": [self.sme_response_],
                "re_context": [self.re_response_],
                "qe_context": [self.qe_response_],
                "asset_description": [self.asset_description_],
                "asset_class": [self.asset_class],
            }
        )
        try:
            df.to_csv(
                f"genai_context_docs_{tmp_asset_init.replace(' ', '')}_{model_initial}_{experiment_id}.csv",
                index=False,
            )
        except:
            pass

        # also store the Agent Prompt
        df = pd.DataFrame(
            {
                "SFAgent_prompt": [self.SFAgent.system_message],
                "SMEAgent_prompt": [self.SMEAgent.system_message],
                "SummarizeAgent_prompt": [self.SummarizeAgent.system_message],
                "QuestionGeneratorAgent_prompt": [
                    self.QuestionGeneratorAgent.system_message
                ],
                "QuestionClassifierAgent_prompt": [
                    self.QuestionClassifierAgent.system_message
                ],
                "QEAgent_prompt": [self.QEAgent.system_message],
                "REAgent_prompt": [self.REAgent.system_message],
            }
        )

        try:
            df.to_csv(
                f"genai_system_prompts_{tmp_asset_init.replace(' ', '')}_{model_initial}_{experiment_id}.csv",
                index=False,
            )
        except:
            pass
