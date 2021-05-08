
## reference from https://github.com/silvanohirtie/youtube-mp3-downloader

from __future__ import unicode_literals
import youtube_dl

def downloadAudio(link):

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
        print("Track downloaded!")
