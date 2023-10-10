from dotenv import load_dotenv
import os
from config.log import log
import openai
from common.one_shot import eg_original_article, eg_summarized_article
from config.env import OPENAI_API_BASE, OPENAI_API_VERSION, OPENAI_API_DEPLOYMENT_NAME, OPENAI_API_TYPE, OPENAI_API_KEY

load_dotenv()

openai.api_type = OPENAI_API_TYPE
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE
openai.api_version = OPENAI_API_VERSION
deployment = OPENAI_API_DEPLOYMENT_NAME

def contextual_prediction(content: list, query: str):

    # Skip for perfect similarity
    labels = set(item["label"] for item in content)
    if len(labels) == 1:
        return labels.pop()

    for i in content:
        res = summarize(text=i['text'])
        i['text'] = res

    article = summarize(text=query)
    
    message = {
        "role": "user",
        "content": f"""
        Content: {content[0]['text']}
        Label: {content[0]['label']}
        
        Content: {content[1]['text']}
        Label: {content[1]['label']}

        Content: {content[2]['text']}
        Label: {content[2]['label']}

        Content: {article}
        Label:
        """
    }

    response = openai.ChatCompletion.create(
        engine=deployment,
        messages=[message],
    )

    if "choices" in response and response["choices"]:
        message = response["choices"][0]["message"]
        if "content" in message:
            result = message["content"]
            return result
        else:
            return "0"

@staticmethod
def summarize(text: str, chars_count: int = 800):
    message = {
        "role": "user",
        "content": f"""
        Summarize this Indonesian news article into {chars_count} characters (output the result in Indonesian language)

        Article: {eg_original_article}
        Summary: {eg_summarized_article}
        
        Article: {text}
        Summary:
        """
    }

    response = openai.ChatCompletion.create(
        engine=deployment,
        messages=[message],
    )

    if "choices" in response and response["choices"]:
        message = response["choices"][0]["message"]
        if "content" in message:
            result = message["content"]
            return result
        else:
            return text