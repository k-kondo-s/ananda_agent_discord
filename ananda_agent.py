# %%
from tools.access_summarization import AccessSummarization
from tools.wallclock import WallClock
from tools.self_ask import SelfAsk
from langchain.agents import Tool, initialize_agent, ConversationalAgent
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI


class AnandaAgent():
    def __init__(self,
                 asum=AccessSummarization(),
                 clock=WallClock(),
                 self_ask=SelfAsk()):
        """AnandaAgent"""
        self.asum = asum
        self.clock = clock
        self.self_ask = self_ask

        self.agent_chain = None

    def get_tool(self):
        """tool を取得"""
        tools = [
            Tool(
                name="AccessAndSummarizeIfAsked",
                func=self.asum.run,
                description="""与えられた URL にアクセスして、内容を要約することができる。
                The input to this tool should be a URL string."""
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
    result = agent.run('要約して https://note.com/kondokenji/n/n9b4cc9c052ef')
    print(result)
