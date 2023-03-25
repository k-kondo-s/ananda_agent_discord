# %%
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper


class SelfAsk():
    def __init__(self,
                 search=GoogleSearchAPIWrapper(k=5)):
        """AnandaAgent"""
        self.search = search

        self.agent_chain = None

    def get_tool(self):
        """tool を取得"""
        tools = [
            # self ask 用の機能
            Tool(
                name="Intermediate Answer",
                func=self.search.run,
                description="検索することができる。"
            )
        ]
        return tools

    def make_llm(self):
        llm = ChatOpenAI(temperature=0.9, model_name='gpt-4')
        return llm

    def make_memory(self):
        """memory を作成"""
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=1, return_messages=True)
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
            tools=tools, llm=llm, agent="self-ask-with-search", verbose=True, memory=memory)

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
    agent = SelfAsk()
    agent.run("今の総理大臣の生年月日は？")
