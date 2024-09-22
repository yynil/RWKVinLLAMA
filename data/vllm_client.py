from openai import OpenAI


# Modify OpenAI's API key and API base to use vLLM's API server.

openai_api_key = "EMPTY"

openai_api_base = "http://localhost:8000/v1"
model_id = '/home/yueyulin/models/Qwen2.5-7B-Instruct/'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
eos_id = tokenizer.eos_token_id
eos_token = tokenizer.eos_token
client = OpenAI(

    # defaults to os.environ.get("OPENAI_API_KEY")

    api_key=openai_api_key,

    base_url=openai_api_base,

)


models = client.models.list()

model = models.data[0].id
print(model)

conversations = [
        "Can you discuss the influence of Persian literature on Jahangir's works?",
        "As an AI language model, I don't have personal opinions, but I can provide you relevant information to the question asked.\n\nJahangir was heavily influenced by Persian literature, and it is evident in his works. He was deeply interested in Persian poetry and painting, and his court was home to some of the most eminent Persian poets of the time. During his reign, the Mughal court became a major patron of Persian literature and arts.\n\nJahangir's own writings are heavily influenced by Persian literature. His journal, the Tuzuk-i-Jahangiri, which he kept throughout his reign, is written in Persian and reflects his knowledge and appreciation of Persian poetry and literature. The Tuzuk includes accounts of royal hunting expeditions, diplomatic missions, and court ceremonies, as well as observations on everyday life and religious matters.\n\nAdditionally, Jahangir's patronage of Persian literature and arts contributed to the development of a distinct Mughal style, which combined Persian influences with indigenous Indian practices. Mughal painting, for example, drew heavily on Persian miniature painting traditions but also incorporated elements of Indian architecture, costume, and landscape.\n\nIn conclusion, Persian literature had a significant influence on Jahangir's works. His interest in Persian poetry and painting, as well as his patronage of Persian literature and arts, helped to shape his own writing and contributed to the development of the Mughal style.",
        "Wow, I didn't realize how much Jahangir was influenced by Persian literature. Do you have any examples of specific works he created that were heavily influenced by Persian literature?",
        "Yes, Jahangir created many works that were influenced by Persian literature. Some of the most notable examples are as follows:\n\n1. The Tuzuk-i-Jahangiri: This is a journal that Jahangir kept throughout his reign, which is written in Persian language. The journal includes accounts of royal missions, political alliances, and insights on everyday life with his commentary on Persian poetry and literature.\n\n2. Jahangirnama: Jahangirnama is an autobiographical account of Jahangir's life, also written in Persian. It is a comprehensive work that describes Jahangir's daily life, court proceedings, and military campaigns.\n\n3. Jahangir's patronage of Persian literature and arts: Jahangir continued the tradition of patronage of Persian literature and painting that was developed by his father, Akbar. During Jahangir's reign, the Mughal court became a major patron of Persian literature, and poets such as Faizi, Faiz, and Qudsi were invited to his court.\n\n4. Jahangir's coins: Jahangir's coins are also influenced by Persian literature. Some of his coins feature verses from Persian poetry, such as Hafiz and Sa'di.\n\nIn conclusion, Jahangir created many works that were heavily influenced by Persian literature. The Tuzuk-i-Jahangiri, Jahangirnama, Jahangir's patronage of Persian literature and arts, and Jahangir's coins are all examples of ways in which Persian literature influenced Jahangir's works.",
        "It's fascinating how Jahangir was not only influenced by Persian literature but also a patron of it. Did his patronage have any significant impact on the development of Persian literature?",
        "Yes, Jahangir\u2019s patronage had a significant impact on the development of Persian literature. During Jahangir's reign, the Mughal court became a major center for Persian poetry and literature, and many of the greatest poets of the time were invited to his court. This led to a remarkable flourishing of Persian literature in India, which ultimately influenced the development of Persian literature in the wider Persian-speaking world.\n\nJahangir\u2019s patronage was especially crucial for poets who were living in Persia, which was then ruled by the Safavid dynasty. The Safavid era was marked by political instability and religious repression, and many poets fled the country. The Mughal court provided a safe haven for them, and poets such as Mirza Salman Savaji and Mirza Abdul Qadir Bedil found support and patronage there. As a result, the Mughal court became an important center for Persian literature and played a major role in the transmission of Persian literary culture.\n\nJahangir himself was also a poet and a writer, and he was deeply knowledgeable about Persian literature. His patronage contributed to the development of a distinct Mughal style, which blended Persian and Indian elements. This style continued to flourish under Jahangir's successors, Shah Jahan and Aurangzeb, and had a lasting impact on the development of Persian literature in India.\n\nIn conclusion, Jahangir\u2019s patronage played a crucial role in the development of Persian literature. The Mughal court became a major center for Persian literature, and Jahangir's support for Persian poets and writers contributed to the flourishing of Persian literary culture in India.",
        "That's amazing how Jahangir's patronage of Persian literature had such a significant impact on its development. I would love to learn more about the Mughal style and how it blended Persian and Indian elements.",
        "Sure, I would be happy to provide some more information on the Mughal style and how it blended Persian and Indian elements.\n\nThe Mughal style is a syncretic style of art that emerged in northern India during the Mughal Empire (1526-1858). The Mughal rulers were of Central Asian origin, and their empire was marked by a rich cultural exchange between Central Asia, Persia, and India.\n\nThe Mughals' patronage of Persian literature and arts led to the development of a distinct Mughal style that combined Persian and Indian elements. In painting, for example, Mughal artists adopted the Persian tradition of miniature painting, but also incorporated Indian elements such as the use of bright colors, detailed depictions of nature, and the portrayal of traditional Indian attire.\n\nThe Mughal style is also evident in architecture, where it combined Persian and Indian styles to create a unique blend of Islamic and Hindu architecture. The Taj Mahal, for example, combines Persian and Central Asian architectural elements with Indian decorative motifs.\n\nMughal music also blended Persian and Indian elements, with the development of a new genre of Indian classical music, known as the \"Mughal school.\" This style incorporated Persian musical elements such as the use of stringed instruments, as well as Indian elements such as the use of the tabla.\n\nIn conclusion, the Mughal style is a syncretic style of art that emerged in northern India during the Mughal Empire. It combined Persian and Indian elements, resulting in a unique blend of Islamic and Hindu architecture, painting, music, and literature.",
        "It's fascinating how much the Mughal style blended elements from different cultures. Do you know of any other cultures that had similar syncretic styles?",
        "Yes, there are several other cultures that have similar syncretic styles that blend elements from different cultures. Here are a few examples:\n\n1. Andalusian art and architecture: Andalusia is a region in southern Spain that was ruled by Muslim emirs from the 8th to the 15th century. During this period, a unique style of art and architecture emerged that blended Islamic and Christian elements. Examples of this include the Alhambra palace in Granada, which features Islamic motifs, but also has architecture elements from earlier Spanish rulers.\n\n2. Indo-Saracenic architecture: Indo-Saracenic architecture is a style that emerged during the British colonial period in India, which blends Indian, Islamic, and European styles. It features elements such as Gothic arches, Islamic domes, and Indian decorative motifs.\n\n3. Chinoiserie: Chinoiserie is a style that emerged in Europe in the 17th and 18th centuries that incorporates elements from Chinese art and design. This includes the use of Chinese decorative motifs, porcelain, and lacquer.\n\n4. Nihonga: Nihonga is a style of Japanese painting that emerged in the late 19th century, which combines traditional Japanese painting techniques with Western painting styles. It features elements such as bright colors, detailed depictions of nature, and historical Japanese themes.\n\nIn conclusion, there are several other cultures that have similar syncretic styles to the Mughal style, which blend elements from different cultures. Andalusian art and architecture, Indo-Saracenic architecture, Chinoiserie, and Nihonga are just a few examples of syncretic styles that have emerged in different cultures throughout history."
    ]
def custom_chat_template_qwen(messages):
    template = ""
    for msg in messages:
        if msg["role"] == "user":
            template += f"<|im_start|>user\n\n{msg['content']}<|im_end|>"
        elif msg["role"] == "assistant":
            template += f"<|im_start|>assistant\n\n{msg['content']}<|im_end|>"
    # Add the final assistant prompt
    template += "<|im_start|>assistant\n\n"
    return template
chats = []
for i in range(0, len(conversations), 2):
    chats.append({
            "role": "user",
            "content": conversations[i]
        })
    prompt = custom_chat_template_qwen(chats)
    completion = client.completions.create(
            
            model=model,
    
            prompt=prompt,
    
            echo=False,
            stop=[eos_token],
            n=1,
            max_tokens=2048,
            stream=False,
            logprobs=1,
            temperature=0.0,
            top_p = 1,
            
        )
    print(completion.usage.completion_tokens)
    generated_text = completion.choices[0].text
    chats.append({
        "role": "assistant",
        "content": generated_text
    })
    current_length = completion.usage.total_tokens
    print(current_length)
    print('-------------------')

import json
print(json.dumps(chats, indent=4))