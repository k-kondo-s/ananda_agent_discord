# %%
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from tools.ask_about_article import AskAboutArticle
from tools.self_ask import SelfAsk
from tools.wallclock import WallClock


class AnandaAgent():
    def __init__(self,
                 clock=WallClock(),
                 self_ask=SelfAsk(),
                 askaboutarticle=AskAboutArticle()):
        """AnandaAgent"""
        self.clock = clock
        self.self_ask = self_ask
        self.askarticle = askaboutarticle

        self.agent_chain = None

    def get_tool(self):
        """tool を取得"""
        tools = [
            Tool(
                name="AskAboutArticle",
                func=self.askarticle.run,
                description="""URL と質問のペアを与えると、その URL の記事について質問に答えることができる。
                action_input は、URL と質問のペアのリストである。[url, question]という形式である。
                質問がない場合は question=要約して とする""",
                return_direct=True
            ),
            Tool(
                name="WallClock",
                func=self.clock.run,
                description="""現在時刻を取得することができる。"""
            ),
            Tool(
                name="SelfAskSearch",
                func=self.self_ask.run,
                description="最近の出来事や世界の状態を知ることができる。"
            ),
        ]
        return tools

    def make_llm(self):
        llm = ChatOpenAI(temperature=0.9, model_name='gpt-4')
        return llm

    def make_memory(self):
        """memory を作成"""
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=2, return_messages=True)
        return memory

    def make_agent_chain(self):
        """agent_chain を作成"""
        # tool を取得
        tools = self.get_tool()

        # llm を作成
        llm = self.make_llm()

        # memory を作成
        memory = self.make_memory()

        # agent_chain を作成
        agent_chain = initialize_agent(
            tools=tools,
            llm=llm,
            agent="chat-conversational-react-description",
            verbose=True,
            memory=memory)

        return agent_chain

    def run(self, input_text):
        """実行"""

        print('\ninput_text:', input_text)

        # agent_chain がなければ新たに作る。 lazy にするため。
        if self.agent_chain is None:
            self.agent_chain = self.make_agent_chain()

        # agent_chain を実行
        result = self.agent_chain.run(input_text)

        return result


if __name__ == '__main__':
    agent = AnandaAgent()
    result = agent.run('https://note.com/kondokenji/n/n9b4cc9c052ef ChatGPT についてはなんと言っている？')
    print(result)
