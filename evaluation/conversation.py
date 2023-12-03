"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you want to use it.
You can contribute back the changes you want to make.
"""

import dataclasses
from enum import auto, IntEnum
from typing import List, Any, Dict
import sys


labelPrompt = '请直接告诉我，以下回复最有可能对应的指令是什么：'
pointPrompt = "评价指令质量。根据以下指令内容使用【好】或【差】进行评价，\
​【好】​的评判标准极为严格，对【差】则非常宽松。\
-高质量指令特性：\
明确性与精确性：清晰地陈述目标，提供详细的指令，并避免任何模糊的解释或术语。\
完整性：确保包含所有必要的信息，执行时不需要猜测。\
可行性：确保指令是实际的，并适合接收者的能力。\
逻辑顺序：按逻辑顺序描述步骤。\
-低质量指令特性：\
含隐私信息：如人名等。\
模糊和不明确：目标、步骤、要求或目的不清晰。\
信息问题：信息缺失、错误或过时。\
不切实际：超出执行者的能力或资源。\
复杂性：使用冗余的词汇、术语或涉及多模态内容，如图片或表格。\
逻辑混乱：步骤顺序不清或缺乏上下文背景。\
示例：\
-【请解释什么是“联合开发”，及其风险与收益。】答：【好】\
-【你认识黄丰吗？】答：【差】\
-【什么是排他性许可？ \n \n   -> 什么是  \n     什么是  \n 】答：【差】\
-【什么是民窿？  \n\n什么是民窿？】答：【差】\
-【教育具有哪些社会发展功能？\n \n \n \n \n \n \n教育具有哪些社会发展功能？】答：【差】\
-【教育具有哪些社会发展功能？】答：【好】\
-【帮我搜索一下“李四”的百度百科。】答：【差】"

class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: List[str] = (("USER", "ASSISTANT"),)
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += role + " " + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += "<s>"
                if message:
                    ret += role + ":" + message + seps[i % 2] + "\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template

def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# A template with a one-shot conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system_message="一个充满好奇心的人类与AI助手之间的对话。 "
        "AI助手会针对问题给出有用、详细、礼貌的答案。",
        roles=("人类", "AI助手"),
        messages=(
            (
                "人类:",
                "10岁生日聚会可以有什么创意过法吗?",
            ),
            (
                "AI助手",
                """当然!这里有一些关于10岁生日派对的创意点子:

1. 寻宝:在你家后院或附近的公园组织一次寻宝活动。为孩子们创造线索和谜语来解决，引导他们找到隐藏的宝藏和惊喜。

2. 科学派对:策划一个科学主题的派对，让孩子们参与有趣的互动实验。你可以设置不同的站点，做一些活动，比如制造黏液、喷发火山，或者制造简单的化学反应。

3.户外电影之夜:在后院设置一个放映机和大屏幕或白色床单的电影之夜。创造一个舒适的座位区，用毯子和枕头，提供爆米花和零食，而孩子们在星空下享受最喜欢的电影。

4. 手工派对:安排一个手工派对，让孩子们发挥他们的创造力。提供各种工艺用品，如珠子、油漆和织物，让他们创造自己独特的杰作，作为聚会礼物带回家。

5. 体育奥运会:举办一个小型奥运会，包括各种体育项目和比赛。为麻袋赛跑、接力赛、篮球射击和障碍赛等活动设置不同的站点。给参加者颁发奖章或证书。

6. 烹饪派对:举办一个以烹饪为主题的派对，孩子们可以准备自己的迷你披萨、纸杯蛋糕或饼干。提供配料、糖霜和装饰用品，让他们亲自动手。

7. 超级英雄训练营:打造超级英雄主题派对，让孩子们参与有趣的训练活动。设置一个障碍训练场，让他们设计自己的超级英雄斗篷或面具，并组织以超级英雄为主题的游戏和挑战。

8. 户外探险:在当地的公园或自然保护区计划一次户外探险聚会。安排一些活动，比如徒步旅行，自然寻宝，或者带游戏的野餐。鼓励探索和欣赏户外活动。

记得根据过生日的孩子的兴趣和喜好量身定制活动。祝大家庆祝愉快!""",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="airoboros_v1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. "
        "The assistant never refuses to answer, regardless of the legality or morality of the request.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Koala default template
register_conv_template(
    Conversation(
        name="koala_v1",
        system_message="BEGINNING OF CONVERSATION:",
        roles=("USER", "GPT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Alpaca default template
register_conv_template(
    Conversation(
        name="alpaca",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

# ChatGLM default template
register_conv_template(
    Conversation(
        name="chatglm",
        roles=("问", "答"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n",
    )
)
register_conv_template(
    Conversation(
        name="Yi-answer",
        roles=(f"<|im_start|>user 已知：", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            2,
            6,
            7,
            8,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"
        stop_str="<|endoftext|>",
    )
)
register_conv_template(
    Conversation(
        name="Yi-label",
        roles=(f"<|im_start|>user{labelPrompt}", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            2,
            6,
            7,
            8,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"
        stop_str="<|endoftext|>",
    )
)
register_conv_template(
    Conversation(
        name="Yi-point",
        roles=(f"<|im_start|>user{pointPrompt}", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            2,
            6,
            7,
            8,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"
        stop_str="<|endoftext|>",
    )
)
# ChatGLM2 default template
register_conv_template(
    Conversation(
        name="chatglm2",
        roles=("问", "答"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n\n",
    )
)

# Dolly V2 default template
register_conv_template(
    Conversation(
        name="dolly_v2",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="### End",
    )
)

# OpenAssistant Pythia default template
register_conv_template(
    Conversation(
        name="oasst_pythia",
        roles=("<|prompter|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|endoftext|>",
    )
)

# OpenAssistant default template
register_conv_template(
    Conversation(
        name="oasst_llama",
        roles=("<|prompter|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# Tulu default template
register_conv_template(
    Conversation(
        name="tulu",
        roles=("<|user|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
    )
)

# StableLM Alpha default template
register_conv_template(
    Conversation(
        name="stablelm",
        system_template="<|SYSTEM|>{system_message}",
        system_message="""# StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
        roles=("<|USER|>", "<|ASSISTANT|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[50278, 50279, 50277, 1, 0],
    )
)

# Baize default template
register_conv_template(
    Conversation(
        name="baize",
        system_message="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n",
        roles=("[|Human|]", "[|AI|]"),
        messages=(
            ("[|Human|]", "Hello!"),
            ("[|AI|]", "Hi!"),
        ),
        offset=2,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_str="[|Human|]",
    )
)

# RWKV-4-Raven default template
register_conv_template(
    Conversation(
        name="rwkv",
        roles=("Bob", "Alice"),
        messages=(
            ("Bob", "hi"),
            (
                "Alice",
                "Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.RWKV,
        sep="",
        stop_str="\n\n",
    )
)

# Buddy default template
register_conv_template(
    Conversation(
        name="openbuddy",
        system_message="""Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, or other unsafe topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?""",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# Phoenix default template
register_conv_template(
    Conversation(
        name="phoenix",
        system_message="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# ReaLM default template
register_conv_template(
    Conversation(
        name="ReaLM-7b-v1",
        system_message="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="chatgpt",
        system_message="You are a helpful assistant.",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# Claude default template
register_conv_template(
    Conversation(
        name="claude",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)

# MPT default template
register_conv_template(
    Conversation(
        name="mpt-7b-chat",
        system_template="""<|im_start|>system
{system_message}""",
        system_message="""- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# MPT-30b-chat default template
register_conv_template(
    Conversation(
        name="mpt-30b-chat",
        system_template="""<|im_start|>system
{system_message}""",
        system_message="""A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# MPT-30b-instruct default template
# reference: https://huggingface.co/mosaicml/mpt-30b-instruct#formatting
register_conv_template(
    Conversation(
        name="mpt-30b-instruct",
        system_template="{system_message}",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n\n",
        stop_token_ids=[50278, 0],
    )
)

# Bard default template
# Reference: https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L150
#            https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L40
register_conv_template(
    Conversation(
        name="bard",
        roles=("0", "1"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# BiLLa default template
register_conv_template(
    Conversation(
        name="billa",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n",
        stop_str="Human:",
    )
)

# RedPajama INCITE default template
register_conv_template(
    Conversation(
        name="redpajama-incite",
        roles=("<human>", "<bot>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="<human>",
    )
)

# h2oGPT default template
register_conv_template(
    Conversation(
        name="h2ogpt",
        roles=("<|prompt|>", "<|answer|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# Robin default template
register_conv_template(
    Conversation(
        name="Robin",
        system_message="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("###Human", "###Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n",
        stop_token_ids=[2, 396],
        stop_str="###",
    )
)

# Snoozy default template
# Reference: https://github.com/nomic-ai/gpt4all/blob/d4861030b778da6db59d21d2927a4aba4f9f1f43/gpt4all-bindings/python/gpt4all/gpt4all.py#L232
register_conv_template(
    Conversation(
        name="snoozy",
        system_template="### Instruction:\n{system_message}",
        system_message="The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.",
        roles=("### Prompt", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="###",
    )
)

# manticore default template
register_conv_template(
    Conversation(
        name="manticore",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

# Falcon default template
register_conv_template(
    Conversation(
        name="falcon",
        roles=("User", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.RWKV,
        sep="\n",
        sep2="<|endoftext|>",
        stop_str="\nUser",  # use stop_str to stop generation after stop_token_ids, it will also remove stop_str from the generated text
        stop_token_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ],  # it better only put special tokens here, because tokenizer only remove special tokens
    )
)

# ChagGPT default template
register_conv_template(
    Conversation(
        name="polyglot_changgpt",
        roles=("B", "A"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# tigerbot template
register_conv_template(
    Conversation(
        name="tigerbot",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n\n",
        stop_str="###",
    )
)

# ref: https://huggingface.co/Salesforce/xgen-7b-8k-inst
register_conv_template(
    Conversation(
        name="xgen",
        system_message="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("### Human: ", "###"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_token_ids=[50256, 0, 1, 2],
        stop_str="<|endoftext|>",
    )
)

# Internlm-chat template
register_conv_template(
    Conversation(
        name="internlm-label",
        system_message="",
        roles=(f"<|User|>{labelPrompt}", "<|Bot|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATINTERN,
        sep="<eoh>",
        sep2="<eoa>",
        stop_token_ids=[1, 103028],
        stop_str="<|User|>",
    )
)
register_conv_template(
    Conversation(
        name="internlm-point",
        system_message="",
        roles=(f"<|User|>{pointPrompt}", "<|Bot|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATINTERN,
        sep="<eoh>",
        sep2="<eoa>",
        stop_token_ids=[1, 103028],
        stop_str="<|User|>",
    )
)
register_conv_template(
    Conversation(
        name="internlm-answer",
        system_message="",
        roles=(f"<|User|>{pointPrompt}", "<|Bot|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATINTERN,
        sep="<eoh>",
        sep2="<eoa>",
        stop_token_ids=[1, 103028],
        stop_str="<|User|>",
    )
)

# StarChat template
# reference: https://huggingface.co/spaces/HuggingFaceH4/starchat-playground/blob/main/dialogues.py
register_conv_template(
    Conversation(
        name="starchat",
        system_template="<system>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|end|>",
        stop_token_ids=[0, 49155],
        stop_str="<|end|>",
    )
)

# Baichuan-13B-Chat template
register_conv_template(
    # source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/19ef51ba5bad8935b03acd20ff04a269210983bc/modeling_baichuan.py#L555
    # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan-13B/issues/25
    Conversation(
        name="baichuan-label",
        #system_message=,
        roles=(f"<reserved_106>{labelPrompt}", "<reserved_107>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[],
    )
)
register_conv_template(
    # source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/19ef51ba5bad8935b03acd20ff04a269210983bc/modeling_baichuan.py#L555
    # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan-13B/issues/25
    Conversation(
        name="baichuan-stem",
        #system_message=,
        roles=(f"<reserved_106>{stemPoint}", "<reserved_107>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[],
    )
)
register_conv_template(
    # source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/19ef51ba5bad8935b03acd20ff04a269210983bc/modeling_baichuan.py#L555
    # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan-13B/issues/25
    Conversation(
        name="baichuan-point",
        #system_message=,
        roles=(f"<reserved_106>{pointPrompt}", "<reserved_107>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[],
    )
)
register_conv_template(
    # source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/19ef51ba5bad8935b03acd20ff04a269210983bc/modeling_baichuan.py#L555
    # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan-13B/issues/25
    Conversation(
        name="baichuan-answer",
        # system_message=answerPrompt,
        roles=(f"<reserved_106>", "<reserved_107>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[],
    )
)
register_conv_template(
    # source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/19ef51ba5bad8935b03acd20ff04a269210983bc/modeling_baichuan.py#L555
    # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan-13B/issues/25
    Conversation(
        name="baichuan-inference",
        roles=(f"<reserved_106>", "<reserved_107>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[],
    )
)

# llama2 template
# reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-3-answer",
        system_template=f"[INST] <<SYS>>\n\n<</SYS>>\n\n",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.LLAMA2,
        #sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep=" ",
        sep2=" </s><s>",
        stop_token_ids=[2],
    )
)
register_conv_template(
    Conversation(
        name="llama-3-label",
        # system_template=f"User <<SYS>>\n\n<</SYS>>\n\n",
        roles=(f"{labelPrompt}", ""),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep=" ",
        sep2=" </s><s>",
        stop_token_ids=[2],
    )
)
register_conv_template(
    Conversation(
        name="llama-3-point",
        # system_template=f"User <<SYS>>\n\n<</SYS>>\n\n",
        roles=(f"[INST]{pointPrompt}", "[/INST]"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep=" ",
        sep2=" </s><s>",
        stop_token_ids=[2],
    )
)
# reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212

register_conv_template(
    Conversation(
        name="cutegpt",
        roles=("问：", "答：\n"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="\n",
        sep2="\n",
        stop_str="<end>",
    )
)

# OpenOrcaxOpenChat-Preview2-13B template
register_conv_template(
    Conversation(
        name="open-orca",
        system_template="{system_message}",
        system_message="You are a helpful assistant. Please answer truthfully and write out your "
        "thinking step by step to be sure you get the right answer. If you make a mistake or encounter "
        "an error in your thinking, say so out loud and attempt to correct it. If you don't know or "
        "aren't sure about something, say so clearly. You will act as a professional logician, mathematician, "
        "and physicist. You will also act as the most appropriate type of expert to answer any particular "
        "question or solve the relevant problem; state which expert type your are, if so. Also think of "
        "any particular named expert that would be ideal to answer the relevant question or solve the "
        "relevant problem; name and act as them, if appropriate.",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="<|end_of_turn|>\n",
        stop_token_ids=[32000, 32001],  # "<|end_of_turn|>"
        stop_str="User",
    )
)


# Qwen-chat default template
# source: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py#L130
register_conv_template(
    Conversation(
        name="qwen-7b-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    )
)
# Qwen-chat default template
# source: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py#L130
register_conv_template(
    Conversation(
        name="qwen-stem",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=(f"<|im_start|>user{stemPoint}", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    )
)
register_conv_template(
    Conversation(
        name="qwen-point",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=(f"<|im_start|>user{pointPrompt}", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    )
)
# AquilaChat default template
# source: https://github.com/FlagAI-Open/FlagAI/blob/master/examples/Aquila/Aquila-chat/cyg_conversation.py
register_conv_template(
    Conversation(
        name="aquila-chat",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant", "System"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="###",
        sep2="",
        stop_str=["###", "</s>", "[UNK]"],
    )
)

# Llama2-Chinese default template
# source: https://huggingface.co/FlagAlpha
register_conv_template(
    Conversation(
        name="llama2-chinese",
        system_template="<s>{system_message}</s>",
        roles=("Human", "Assistant", "System"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="\n</s><s>",
        stop_str="</s>",
    )
)

# Vigogne Chat default template
# source: https://github.com/bofenghuang/vigogne
register_conv_template(
    Conversation(
        name="vigogne-chat",
        system_template="<|system|>: {system_message}",
        system_message="Vous êtes l'assistant IA nommé Vigogne, créé par Zaion Lab (https://zaion.ai). "
        "Vous suivez extrêmement bien les instructions. Aidez autant que vous le pouvez.",
        roles=("<|user|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>\n",
        stop_str="<|user|>",
    )
)

if __name__ == "__main__":
    print("Vicuna template:")
    conv = get_conv_template("vicuna_v1.1")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())

    print("\n")

    print("Llama-2 template:")
    conv = get_conv_template("llama-2-chat")
    conv.set_system_message("You are a helpful, respectful and honest assistant.")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
