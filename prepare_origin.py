import os
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

def save_origin(data_path='./data'):
   train_df = pd.read_csv(data_path+'/train.csv')
   for index in tqdm(range(len(train_df))):
      sample_df = train_df.iloc[index]

      # train 이미지 불러오기
      train_path = sample_df['img_path'].split('/')[-1]
      train_img = Image.open(data_path+'/train/'+train_path)
      
      width, height = train_img.size
      cell_width = width // 4
      cell_height = height // 4

      # 4x4 이미지 행렬 생성
      origin_img = Image.new("RGB", (width, height))
      numbers = list(sample_df)[2:]

      # 정렬된 이미지 생성 및 저장
      i = 0
      for row in range(4):
         for col in range(4):

            def get_cell_info(row, col):
               left = col * cell_width
               right = left + cell_width
               upper = row * cell_height
               lower = upper + cell_height
               return left, upper, right, lower
            
            # 부분 이미지 추출
            tile_loc = get_cell_info(row, col)
            tile = train_img.crop(tile_loc)

            # 부분 이미지를 4x4 행렬 위치에 합성
            origin_tile_loc = get_cell_info((numbers[i] - 1) // 4, (numbers[i] - 1) % 4)
            origin_img.paste(tile, origin_tile_loc)
            i += 1

      # 재정렬된 이미지 저장
      if not os.path.exists('./data/origin'):
         os.mkdir('./data/origin/')
      origin_name = f'ORIGIN_{index:05}.jpg'
      origin_path = './data/origin/'+origin_name
      origin_img.save(origin_path)