# %%
from access_summarization import AccessSummarization
from wallclock import WallClock
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper


class AnandaAgent():
    def __init__(self,
                 asum=AccessSummarization(),
                 search=GoogleSearchAPIWrapper(),
                 clock=WallClock()):
        """AnandaAgent"""
        self.asum = asum
        self.search = search
        self.clock = clock

        self.agent_chain = None

    def get_tool(self):
        """tool を取得"""
        tools = [
            Tool(
                name="AccessAndSummarizeIfAsked",
                func=self.asum.run,
                description="""使用できない。与えられた URL を要約するように指示があった場合にのみ使用できる。
                The input to this tool should be a URL string."""
            ),
            Tool(
                name="Search",
                func=self.search.run,
                description="""最近の出来事や知らないことなど、検索して調べたいときに使用する。
                """
            ),
            Tool(
                name="WallClock",
                func=self.clock.run,
                description="""現在時刻を取得したいときに使用する。"""
            )
            # # self ask 用の機能
            # Tool(
            #     name="Intermediate Answer",
            #     func=self.search.run,
            #     description="useful for when you need to ask with search"
            # )
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
        # agent には chat-conversational-react-description, self-ask-with-search などがある。
        agent_chain = initialize_agent(
            tools=tools, llm=llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
        # tools=tools, llm=llm, agent="self-ask-with-search", verbose=True, memory=memory)

        return agent_chain

    def run(self, input_text):
        """実行"""

        print('input_text:', input_text)

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
