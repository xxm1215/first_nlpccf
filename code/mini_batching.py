import os

def count(folder):
    img = ['.jpg','.jpeg','.png']
    c = 0

    for root,dirs,files in os.walk(folder):
        for file in files:
            _,extension = os.path.splitext(file)
            if extension.lower() in img:
                c+=1

    return c

folder='/media/qust521/qust_3_big/fake_news_data/weibo/nlpccf/weibo17/rumor_images'

num_image =count(folder)

print(f"{num_image}")