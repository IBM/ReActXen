from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from reactxen.utils.watsonx_llm import granite_llm

def run_multi_step_chain(
    llm=granite_llm,
    system_message=None,
    questions=[],
    history_depth=-1,
    asset_class="Chiller",
    asset_description="this is a closed-loop water-cooler chiller",
):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), ("human", questions[0])]
    )
    chain = prompt | llm | StrOutputParser()
    ans = chain.invoke(
        {
            "asset_class": asset_class,
            "asset_description": asset_description,
        }
    )
    ans = ans.replace('{','{{')
    ans = ans.replace('}','}}')
    for pt in range(1, len(questions)):
        if history_depth == -1:  # slow
            prompt.extend([("ai", ans), ("human", questions[pt])])
        elif history_depth == 0:  # only use the current
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_message), ("human", questions[0])]
            )
        else:
            pass
        chain = prompt | llm | StrOutputParser()
        ans = chain.invoke(
            {
                "asset_class": asset_class,
                "asset_description": asset_description,
            }
        )
        ans = ans.replace('{','{{')
        ans = ans.replace('}','}}')
    return ans
