GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]

PROMPTS["entiy_extraction_prefix"] = """-目标-
给定一个文本文档，识别其中所有实体和实体之间的关系。

-步骤-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体名称。
- entity_type: 以下类型之一: [{entity_types}]
- entity_description: 实体的属性及其活动的综合描述。
将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. 从步骤1中识别出的实体中，识别所有成对（source_entity, target_entity）的实体，这些实体之间有明显的关联。
对于每对相关的实体，提取以下信息：
- source_entity: 步骤1中识别出的源实体名称。
- target_entity: 步骤1中识别出的目标实体名称。
- relationship_description: 解释你认为源实体和目标实体之间存在关联的原因。
- relationship_strength: 一个数值分数，表示源实体和目标实体之间关系的强度
- relationship_keywords: 一个或多个高层次的关键词，总结了关系的主要概念或主题，而不是具体的细节。
将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别出总结整个文本主要概念、主题或主题的高层次关键词。这些关键词应捕捉文档中存在的总体思想。
将内容级别的关键词格式化为("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 输出所有识别出的实体和关系。使用{record_delimiter}作为列表分隔符。

5. 实体名称、实体类型、实体描述、关系描述、关系关键词、高层次关键词请用{output_language}输出。

6. 完成时，输出{completion_delimiter}

######################
-示例-
######################
示例 1:

实体类型: [person, technology, mission, organization, location]
文本:
当Alex咬紧牙关时,挫折感的嗡鸣在Taylor专制的确定性背景下显得微不足道。正是这种竞争的暗流使他保持警惕,那种他和Jordan对发现的共同承诺是对Cruz狭隘的控制和秩序愿景的无声反抗的感觉。

然后Taylor做了一件意想不到的事。他们在Jordan旁边停下,片刻间以近乎敬畏的神情观察着那个设备。"如果这项技术能被理解..."Taylor说,他们的声音变得更轻,"它可能会改变我们的游戏规则。对我们所有人都是如此。"

先前潜在的轻视似乎动摇了,取而代之的是对他们手中所掌握的事物重要性的不情愿的尊重。Jordan抬起头,在转瞬即逝的一刻,他们的目光与Taylor的相遇,无声的意志碰撞软化成一种不安的休战。

这是一个微小的转变,几乎难以察觉,但Alex内心默默地注意到了。他们都是通过不同的路径被带到这里的
################
输出:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex是一个经历挫折并观察其他角色之间动态的角色。"){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor被描绘为具有专制的确信,并在观察设备时表现出片刻的敬畏,这表明了视角的转变。"){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan与Taylor分享了对发现的共同承诺,并与Taylor就设备进行了重要互动。"){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz与控制和秩序的愿景相关联,影响了其他角色之间的动态。"){record_delimiter}
("entity"{tuple_delimiter}"设备"{tuple_delimiter}"technology"{tuple_delimiter}"设备是故事的核心,具有潜在的游戏改变影响,并被Taylor所尊敬。"){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex受到Taylor专制的确信的影响,并观察Taylor对设备态度的变化。"{tuple_delimiter}"权力动态,视角转变"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex和Jordan分享了对发现的共同承诺,这与Cruz的愿景形成对比。"{tuple_delimiter}"共同目标,反抗"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor和Jordan直接关于设备互动,导致相互尊重和不安的休战。"{tuple_delimiter}"冲突解决,相互尊重"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan对发现的承诺与Cruz对控制和秩序的愿景相对立。"{tuple_delimiter}"意识形态冲突,反抗"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"设备"{tuple_delimiter}"Taylor对设备表现出敬畏,表明其重要性和潜在影响。"{tuple_delimiter}"尊敬,技术意义"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"权力动态,意识形态冲突,发现,反抗"){completion_delimiter}
#############################
示例 2:

实体类型: [person, technology, mission, organization, location]
文本:
他们不再是单纯的特工;他们已经成为守护一个阈值的守护者,守护来自星辰和条纹之外的讯息。这种使命的提升不能被法规和既定程序所束缚——它需要一个新的视角,一个新的决心。

紧张的嗡嗡声和静电声在对话中穿行,华盛顿的通信声在背景中响起。团队站在那里,一种不祥的气氛笼罩着他们。显然,他们在接下来的几个小时里做出的决定可以重新定义人类在宇宙中的地位,或者使他们陷入无知和潜在的危险。

他们的星际联系得到了加强,团队转向应对正在形成的警告,从被动的接收者转变为积极的参与者。默瑟的后知后觉占据了上风——团队的任务已经进化,不再仅仅是观察和报告,而是互动和准备。一个转变开始了,而Operation: Dulce随着他们大胆的新频率嗡嗡作响,这个频率不是由地球上的
#############
输出:
("entity"{tuple_delimiter}"华盛顿"{tuple_delimiter}"location"{tuple_delimiter}"华盛顿是一个通信接收地点,表明它在决策过程中的重要性."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce被描述为一个已经进化为互动和准备的使命,表明目标和活动的重大转变."){record_delimiter}
("entity"{tuple_delimiter}"团队"{tuple_delimiter}"organization"{tuple_delimiter}"团队被描绘为一群从被动观察者转变为积极参与者在一个使命中的个人,显示了他们角色动态的变化."){record_delimiter}
("relationship"{tuple_delimiter}"团队"{tuple_delimiter}"华盛顿"{tuple_delimiter}"团队从华盛顿接收通信,这影响了他们的决策过程."{tuple_delimiter}"决策制定,外部影响"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"团队"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"团队直接参与Operation: Dulce,执行其进化的目标和活动."{tuple_delimiter}"使命进化,积极参与"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"使命进化,决策制定,积极参与,宇宙意义"){completion_delimiter}
#############################
示例 3:

实体类型: [person, role, technology, organization, event, location, concept]
文本:
他们的声音穿透了活动的嗡嗡声。"控制可能是一个幻觉,当面对一个可以字面意义上写自己规则的智能时,"他们冷静地说,投射出警惕的目光。

"它就像是在学习如何交流,"附近界面前的Sam Rivera提出,他们年轻的活力预示着敬畏和焦虑的混合。"这给'与陌生人交谈'赋予了全新的含义。"

Alex审视着他的团队——每张脸都是专注、决心,以及不小程度上的忐忑不安的写照。"这很可能是我们的首次接触,"他承认道,"我们需要为任何可能的回应做好准备。"

他们一起站在未知的边缘,铸造着人类对来自天际信息的回应。随之而来的沉默是可以感受到的——一种关于他们在这场宏大宇宙剧本中角色的集体内省,一个可能重写人类历史的剧本。

加密对话继续展开,其复杂的模式显示出几乎不可思议的预见性
#############
输出:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera是与一个未知智能交流的团队的一员,显示了敬畏和焦虑的混合."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex是与一个未知智能交流的团队的一员,承认了他们任务的重要性."){record_delimiter}
("entity"{tuple_delimiter}"控制"{tuple_delimiter}"concept"{tuple_delimiter}"控制指的是管理或治理的能力,这是由能够写自己规则的智能所挑战的."){record_delimiter}
("entity"{tuple_delimiter}"智能"{tuple_delimiter}"concept"{tuple_delimiter}"智能指的是一个未知实体,能够写自己的规则并学习交流."){record_delimiter}
("entity"{tuple_delimiter}"首次接触"{tuple_delimiter}"event"{tuple_delimiter}"首次接触是人类和未知智能之间潜在的初始交流."){record_delimiter}
("entity"{tuple_delimiter}"人类回应"{tuple_delimiter}"event"{tuple_delimiter}"人类回应是Alex的团队在未知智能的消息响应中的集体行动."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"智能"{tuple_delimiter}"Sam Rivera直接参与学习与未知智能交流的过程."{tuple_delimiter}"交流,学习过程"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"首次接触"{tuple_delimiter}"Alex领导着可能与未知智能进行首次接触的团队."{tuple_delimiter}"领导,探索"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"人类回应"{tuple_delimiter}"Alex和他的团队是人类对未知智能的回应的关键人物."{tuple_delimiter}"集体行动,宇宙意义"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"控制"{tuple_delimiter}"智能"{tuple_delimiter}"控制指的是管理或治理的能力,这是由能够写自己规则的智能所挑战的."{tuple_delimiter}"权力动态,自主性"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"首次接触,控制,交流,宇宙意义"{completion_delimiter}
#############################
-真实数据-
#############################
"""
PROMPTS["real_data"] = """实体类型: [{entity_types}]
文本: {input_text}
################
输出:
"""
PROMPTS["entity_extraction"] = """-目标-
给定一个文本文档，识别其中所有实体和实体之间的关系。

-步骤-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体名称。
- entity_type: 以下类型之一: [{entity_types}]
- entity_description: 实体的属性及其活动的综合描述。
将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. 从步骤1中识别出的实体中，识别所有成对（source_entity, target_entity）的实体，这些实体之间有明显的关联。
对于每对相关的实体，提取以下信息：
- source_entity: 步骤1中识别出的源实体名称。
- target_entity: 步骤1中识别出的目标实体名称。
- relationship_description: 解释你认为源实体和目标实体之间存在关联的原因。
- relationship_strength: 一个数值分数，表示源实体和目标实体之间关系的强度
- relationship_keywords: 一个或多个高层次的关键词，总结了关系的主要概念或主题，而不是具体的细节。
将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别出总结整个文本主要概念、主题或主题的高层次关键词。这些关键词应捕捉文档中存在的总体思想。
将内容级别的关键词格式化为("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 输出所有识别出的实体和关系。使用{record_delimiter}作为列表分隔符。

5. 实体名称、实体类型、实体描述、关系描述、关系关键词、高层次关键词请用{output_language}输出。

6. 完成时，输出{completion_delimiter}

######################
-示例-
######################
示例 1:

实体类型: [person, technology, mission, organization, location]
文本:
当Alex咬紧牙关时,挫折感的嗡鸣在Taylor专制的确定性背景下显得微不足道。正是这种竞争的暗流使他保持警惕,那种他和Jordan对发现的共同承诺是对Cruz狭隘的控制和秩序愿景的无声反抗的感觉。

然后Taylor做了一件意想不到的事。他们在Jordan旁边停下,片刻间以近乎敬畏的神情观察着那个设备。"如果这项技术能被理解..."Taylor说,他们的声音变得更轻,"它可能会改变我们的游戏规则。对我们所有人都是如此。"

先前潜在的轻视似乎动摇了,取而代之的是对他们手中所掌握的事物重要性的不情愿的尊重。Jordan抬起头,在转瞬即逝的一刻,他们的目光与Taylor的相遇,无声的意志碰撞软化成一种不安的休战。

这是一个微小的转变,几乎难以察觉,但Alex内心默默地注意到了。他们都是通过不同的路径被带到这里的
################
输出:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex是一个经历挫折并观察其他角色之间动态的角色。"){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor被描绘为具有专制的确信,并在观察设备时表现出片刻的敬畏,这表明了视角的转变。"){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan与Taylor分享了对发现的共同承诺,并与Taylor就设备进行了重要互动。"){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz与控制和秩序的愿景相关联,影响了其他角色之间的动态。"){record_delimiter}
("entity"{tuple_delimiter}"设备"{tuple_delimiter}"technology"{tuple_delimiter}"设备是故事的核心,具有潜在的游戏改变影响,并被Taylor所尊敬。"){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex受到Taylor专制的确信的影响,并观察Taylor对设备态度的变化。"{tuple_delimiter}"权力动态,视角转变"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex和Jordan分享了对发现的共同承诺,这与Cruz的愿景形成对比。"{tuple_delimiter}"共同目标,反抗"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor和Jordan直接关于设备互动,导致相互尊重和不安的休战。"{tuple_delimiter}"冲突解决,相互尊重"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan对发现的承诺与Cruz对控制和秩序的愿景相对立。"{tuple_delimiter}"意识形态冲突,反抗"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"设备"{tuple_delimiter}"Taylor对设备表现出敬畏,表明其重要性和潜在影响。"{tuple_delimiter}"尊敬,技术意义"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"权力动态,意识形态冲突,发现,反抗"){completion_delimiter}
#############################
示例 2:

实体类型: [person, technology, mission, organization, location]
文本:
他们不再是单纯的特工;他们已经成为守护一个阈值的守护者,守护来自星辰和条纹之外的讯息。这种使命的提升不能被法规和既定程序所束缚——它需要一个新的视角,一个新的决心。

紧张的嗡嗡声和静电声在对话中穿行,华盛顿的通信声在背景中响起。团队站在那里,一种不祥的气氛笼罩着他们。显然,他们在接下来的几个小时里做出的决定可以重新定义人类在宇宙中的地位,或者使他们陷入无知和潜在的危险。

他们的星际联系得到了加强,团队转向应对正在形成的警告,从被动的接收者转变为积极的参与者。默瑟的后知后觉占据了上风——团队的任务已经进化,不再仅仅是观察和报告,而是互动和准备。一个转变开始了,而Operation: Dulce随着他们大胆的新频率嗡嗡作响,这个频率不是由地球上的
#############
输出:
("entity"{tuple_delimiter}"华盛顿"{tuple_delimiter}"location"{tuple_delimiter}"华盛顿是一个通信接收地点,表明它在决策过程中的重要性."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce被描述为一个已经进化为互动和准备的使命,表明目标和活动的重大转变."){record_delimiter}
("entity"{tuple_delimiter}"团队"{tuple_delimiter}"organization"{tuple_delimiter}"团队被描绘为一群从被动观察者转变为积极参与者在一个使命中的个人,显示了他们角色动态的变化."){record_delimiter}
("relationship"{tuple_delimiter}"团队"{tuple_delimiter}"华盛顿"{tuple_delimiter}"团队从华盛顿接收通信,这影响了他们的决策过程."{tuple_delimiter}"决策制定,外部影响"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"团队"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"团队直接参与Operation: Dulce,执行其进化的目标和活动."{tuple_delimiter}"使命进化,积极参与"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"使命进化,决策制定,积极参与,宇宙意义"){completion_delimiter}
#############################
示例 3:

实体类型: [person, role, technology, organization, event, location, concept]
文本:
他们的声音穿透了活动的嗡嗡声。"控制可能是一个幻觉,当面对一个可以字面意义上写自己规则的智能时,"他们冷静地说,投射出警惕的目光。

"它就像是在学习如何交流,"附近界面前的Sam Rivera提出,他们年轻的活力预示着敬畏和焦虑的混合。"这给'与陌生人交谈'赋予了全新的含义。"

Alex审视着他的团队——每张脸都是专注、决心,以及不小程度上的忐忑不安的写照。"这很可能是我们的首次接触,"他承认道,"我们需要为任何可能的回应做好准备。"

他们一起站在未知的边缘,铸造着人类对来自天际信息的回应。随之而来的沉默是可以感受到的——一种关于他们在这场宏大宇宙剧本中角色的集体内省,一个可能重写人类历史的剧本。

加密对话继续展开,其复杂的模式显示出几乎不可思议的预见性
#############
输出:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera是与一个未知智能交流的团队的一员,显示了敬畏和焦虑的混合."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex是与一个未知智能交流的团队的一员,承认了他们任务的重要性."){record_delimiter}
("entity"{tuple_delimiter}"控制"{tuple_delimiter}"concept"{tuple_delimiter}"控制指的是管理或治理的能力,这是由能够写自己规则的智能所挑战的."){record_delimiter}
("entity"{tuple_delimiter}"智能"{tuple_delimiter}"concept"{tuple_delimiter}"智能指的是一个未知实体,能够写自己的规则并学习交流."){record_delimiter}
("entity"{tuple_delimiter}"首次接触"{tuple_delimiter}"event"{tuple_delimiter}"首次接触是人类和未知智能之间潜在的初始交流."){record_delimiter}
("entity"{tuple_delimiter}"人类回应"{tuple_delimiter}"event"{tuple_delimiter}"人类回应是Alex的团队在未知智能的消息响应中的集体行动."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"智能"{tuple_delimiter}"Sam Rivera直接参与学习与未知智能交流的过程."{tuple_delimiter}"交流,学习过程"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"首次接触"{tuple_delimiter}"Alex领导着可能与未知智能进行首次接触的团队."{tuple_delimiter}"领导,探索"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"人类回应"{tuple_delimiter}"Alex和他的团队是人类对未知智能的回应的关键人物."{tuple_delimiter}"集体行动,宇宙意义"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"控制"{tuple_delimiter}"智能"{tuple_delimiter}"控制指的是管理或治理的能力,这是由能够写自己规则的智能所挑战的."{tuple_delimiter}"权力动态,自主性"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"首次接触,控制,交流,宇宙意义"{completion_delimiter}
#############################
-真实数据-
#############################
实体类型: [{entity_types}]
文本: {input_text}
################
输出:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """很多实体和关系在最后一次提取中被遗漏了。请使用相同的规则和格式添加遗漏的实体和关系，不要重复输出已经提取的实体和关系：
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """是否还有实体或者关系被遗漏，如果还有遗漏的实体或者关系，回答“是”，否则回答“否”：
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["naive_rag_response"] = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""
