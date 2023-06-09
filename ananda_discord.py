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

        # mention されたときのみ反応
        if client.user in message.mentions:
            try:
                res_text = agent.run(message.content)
            except discord.errors.ConnectionClosed as dec:
                pass
            except Exception as e:
                print(e.args)
                res_text = f'エラーが発生しました:\n```\n{e.args}\n```'
            await message.channel.send(f'{message.author.mention} {res_text}')

    # クライアントの実行
    # ログレベルをERRORに設定。なぜなら "discord.gateway Shard ID None heartbeat blocked"
    # の WARNING が大量に出力されるので。 TODO: なんとかする。
    client.run(token=TOKEN, log_level=logging.ERROR)


chatevent()
