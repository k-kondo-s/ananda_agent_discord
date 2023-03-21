# %%
from ananda_agent import AnandaAgent
import discord
import os
import logging


agent = AnandaAgent()


def chatevent():
    # 環境変数の DISCORD_TOKEN からトークンを取得
    try:
        TOKEN = os.environ['DISCORD_TOKEN']
    except KeyError:
        print('環境変数 DISCORD_TOKEN が設定されていません')
        return

    # インテントの生成
    intents = discord.Intents.default()
    intents.message_content = True

    # クライアントの生成
    client = discord.Client(intents=intents)

    # discordと接続した時に呼ばれる
    @client.event
    async def on_ready():
        print(f'We have logged in as {client.user}')

    # メッセージを受信した時に呼ばれる
    @client.event
    async def on_message(message):

        # 自分のメッセージを無効
        if message.author == client.user:
            return

        # メッセージを返す
        try:
            res_text = agent.run(message.content)
        except discord.errors.ConnectionClosed as dec:
            pass
        except Exception as e:
            print(e.args)
            res_text = 'エラーが発生しました。'
        await message.channel.send(res_text)


    # クライアントの実行
    # ログレベルをERRORに設定。なぜなら "discord.gateway Shard ID None heartbeat blocked"
    # の WARNING が大量に出力されるので。 TODO: なんとかする。
    client.run(token=TOKEN, log_level=logging.ERROR)


chatevent()
