# %%
from datetime import datetime
import pytz


class WallClock():
    """WallClock class
    現在時刻をただ取得する
    """
    def __init__(self) -> None:
        pass

    def run(self, input_text=None):
        """実行"""
        print("WallClock run")
        tz = pytz.timezone('Asia/Tokyo')
        now_str = datetime.now(tz).isoformat()
        return now_str


if __name__ == "__main__":
    wall_clock = WallClock()
    c = wall_clock.run("")
    print(c)
