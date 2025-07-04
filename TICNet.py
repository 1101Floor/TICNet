import tensorflow as tf
import time
import pandas as pd
import numpy as np
# import random
import datetime
from matplotlib import pyplot as plt
import argparse

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0],True)


def min_max_normalize(lat_lon, lat_min, lat_max, lon_min, lon_max
                      , cou_min, cou_max, spd_min, spd_max):
    cou = lat_lon[..., 0]
    spd = lat_lon[..., 1]
    lon = lat_lon[..., 2]
    lat = lat_lon[..., 3]
    
    normalized_cou = (cou - cou_min) / (cou_max - cou_min)
    normalized_spd = (spd - spd_min) / (spd_max - spd_min)
    normalized_lon = (lon - lon_min) / (lon_max - lon_min)
    normalized_lat = (lat - lat_min) / (lat_max - lat_min)   
    return tf.stack([normalized_cou, normalized_spd, normalized_lon, normalized_lat], axis=-1)

def min_max_denormalize(normalized_lat_lon, lat_min, lat_max, lon_min, lon_max):
    normalized_lon = normalized_lat_lon[..., 0]
    normalized_lat = normalized_lat_lon[..., 1]
    
    lon = normalized_lon * (lon_max - lon_min) + lon_min
    lat = normalized_lat * (lat_max - lat_min) + lat_min
    
    return tf.stack([lon, lat], axis=-1)



"""   评估函数       """
# 定义Haversine距离计算函数
def haversine_distance(y_true, y_pred):
    lon1, lat1 = y_true[..., 0], y_true[..., 1]
    lon2, lat2 = y_pred[..., 0], y_pred[..., 1]
    
    # 将角度转换为弧度
    lon1, lat1, lon2, lat2 = map(tf.convert_to_tensor, [lon1, lat1, lon2, lat2])
    lon1, lat1, lon2, lat2 = map(lambda x: x * np.pi / 180.0, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = tf.sin(dlat / 2.0) ** 2 + tf.cos(lat1) * tf.cos(lat2) * tf.sin(dlon / 2.0) ** 2
    c = 2 * tf.asin(tf.sqrt(a))
    
    return c * 6371 * 1000  # 返回距离（单位：米）

# 自定义Haversine距离指标类
class HaversineDistanceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='haversine_distance', **kwargs):
        super(HaversineDistanceMetric, self).__init__(name=name, **kwargs)
        self.total_distance = self.add_weight(name='total_distance', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred
                     , sample_weight=None):
        y_true_denorm = min_max_denormalize(y_true, lat_min, lat_max, lon_min, lon_max)
        y_pred_denorm = min_max_denormalize(y_pred, lat_min, lat_max, lon_min, lon_max)
        # y_true_denorm = Denormalization(y_true, ais_lon_lat.mean(axis=0), ais_lon_lat.std(axis=0))
        # y_pred_denorm = Denormalization(y_pred, ais_lon_lat.mean(axis=0), ais_lon_lat.std(axis=0))
        
        distances = haversine_distance(y_true_denorm, y_pred_denorm)
        # distances = haversine_distance(y_true, y_pred)
        distances = tf.reduce_mean(distances, axis=-1)  # 平均每个样本的10个距离
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
            distances = tf.multiply(distances, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(distances), dtype=tf.float32))
        
        self.total_distance.assign_add(tf.reduce_sum(distances))
    

    def result(self):
        return self.total_distance / self.count

    def reset_state(self):
        self.total_distance.assign(0.)
        self.count.assign(0.)

        
class EuclideanDistanceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='euclidean_distance', **kwargs):
        super(EuclideanDistanceMetric, self).__init__(name=name, **kwargs)
        self.total_distance = self.add_weight(name='total_distance', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 计算每个样本的所有目标位置之间的欧氏距离
        distances = tf.norm(y_true - y_pred, axis=-1)
        
        # 计算每个样本的平均欧氏距离
        mean_distances = tf.reduce_mean(distances, axis=-1)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
            mean_distances = tf.multiply(mean_distances, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(mean_distances), dtype=tf.float32))
        
        self.total_distance.assign_add(tf.reduce_sum(mean_distances))

    def result(self):
        return self.total_distance / self.count

    def reset_state(self):
        self.total_distance.assign(0.)
        self.count.assign(0.)
  
     
class HaversineDistanceMetricFianl(tf.keras.metrics.Metric):
    def __init__(self, name='haversine_distanceFianl', **kwargs):
        super(HaversineDistanceMetricFianl, self).__init__(name=name, **kwargs)
        self.total_distance = self.add_weight(name='total_distance', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 取出每个样本的最后一个经纬度对
        y_true_last = y_true[:, -1, :]
        y_pred_last = y_pred[:, -1, :]
        y_true_denorm = min_max_denormalize(y_true_last, lat_min, lat_max, lon_min, lon_max)
        y_pred_denorm = min_max_denormalize(y_pred_last, lat_min, lat_max, lon_min, lon_max)
        
        distances = haversine_distance(y_true_denorm, y_pred_denorm)
        # distances = haversine_distance(y_true_last, y_pred_last)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
            distances = tf.multiply(distances, sample_weight)
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(distances), dtype=tf.float32))
        
        self.total_distance.assign_add(tf.reduce_sum(distances))

    def result(self):
        return self.total_distance / self.count

    def reset_state(self):
        self.total_distance.assign(0.)
        self.count.assign(0.)

def convert_all_to_dataframe(data_dict):
    i = 0
    ii = 0
    j = 0
    a = 0
    for key in data_dict:
        array_4d = data_dict[key]
        b, c, h, w = array_4d.shape
        reshaped_array = array_4d.reshape(b, -1)
        if (i+1) % 2 == 0:
            d = pd.DataFrame(reshaped_array) 
            all_data = pd.concat([a, d], axis=1) #列合并
            j = 0
            i += 1
        else:
            a = pd.DataFrame(reshaped_array)
            j+=1
            i += 1
        if j == 0: 
            if ii == 0:
                df = all_data.copy()
            else:
                df = pd.concat([df, all_data], axis=0)  #行合并
            ii += 1

    return df

#--------------------------------------#
#                  主程序               #
#-------------------------------------#

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, hid_dim, n_heads):
        """
        初始化多头注意力层。
        参数:
            hid_dim: 输入向量的维度。
            n_heads: 注意力头的数量。
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = hid_dim
        self.num_heads = n_heads
        
        assert hid_dim % n_heads == 0, "d_model 必须能被 num_heads 整除"
        
        # 每个头的维度
        self.depth = hid_dim // n_heads
        
        # 定义线性变换层
        self.Wq = tf.keras.layers.Dense(hid_dim)  # Query 线性变换 #, activation="softmax"
        self.Wk = tf.keras.layers.Dense(hid_dim)  # Key 线性变换 @￥#, activation="softmax"
        self.Wv = tf.keras.layers.Dense(hid_dim)  # Value 线性变换
        self.dense = tf.keras.layers.Dense(hid_dim)  # 输出线性变换
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def split_heads(self, x):
        """
        将输入张量分割为多个头。
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            batch_size: 批次大小
        返回:
            分割后的张量，形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (-1, x.shape[1], self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # 调整维度顺序
    
    def CschSoftmax(self, x):
        csch = tf.math.reciprocal(tf.sinh(x), name="li")
        return csch / tf.reduce_sum(csch, axis=-1, keepdims=True)
    
    def kl_divergence(self, x, y):
        kl_divergence = tf.multiply(tf.expand_dims(x, 3)
            ,tf.math.log(tf.expand_dims(x, 3)) - tf.math.log(tf.expand_dims(y, 2) + 2 * 1e-8)
            )
        return kl_divergence
    
    def ComputeKullbackLeiblerDivergence(self, q, k, v, mask=None):
        """
        计算 KL 散度注意力得分（四维多头版本）
        Args:
            q: 查询张量 (batch_size, num_heads, seq_len_q, depth)
            k: 键张量 (batch_size, num_heads, seq_len_k, depth)  
        Returns:
            kl_scores: KL 分数 (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # 确保 q 和 k 是概率分布
        q_prob = tf.nn.softmax(q, axis=-1)  # (batch_size, num_heads, seq_len_q, depth_q)
        k_prob = tf.nn.softmax(k, axis=-1)  # (batch_size, num_heads, seq_len_k, depth_k)

        # 对特征维度求和得到最终得分
        kl_scores = tf.reduce_sum(self.kl_divergence(q_prob, k_prob), axis=-1)  #(batch_size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)  # 获取键向量的维度
        scaled_attention_logits = kl_scores / tf.math.sqrt(dk * 1.0)

        # 应用掩码（如果存在）
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = self.CschSoftmax(scaled_attention_logits)
        del scaled_attention_logits
        # 加权求和
        output = tf.matmul(attention_weights, v)

        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        缩放点积注意力机制。  
        参数:
            q: Query 张量，形状为 (..., seq_len_q, depth)
            k: Key 张量，形状为 (..., seq_len_k, depth)
            v: Value 张量，形状为 (..., seq_len_v, depth)
            mask: 可选的掩码张量
        返回:
            输出张量和注意力权重
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # 计算 Q 和 K 的点积，形状为 (..., seq_len_q, seq_len_k)
        
        # 缩放点积
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # 应用掩码（如果存在）
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # 计算 Softmax 权重
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # 加权求和
        output = tf.matmul(attention_weights, v)  # 形状为 (..., seq_len_q, depth)
        return output, attention_weights
    
    def call(self, q, k, method="dot", mask=None):
        """
        多头注意力的前向传播。
        参数:
            q: Query 张量，形状为 (batch_size, seq_len_q, d_model)
            k: Key 张量，形状为 (batch_size, seq_len_k, d_model)
            v: Value 张量，形状为 (batch_size, seq_len_v, d_model)
            mask: 可选的掩码张量z
        返回:
            输出张量和注意力权重
        """
        # 线性变换
        q = self.Wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.Wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.Wv(k)  # (batch_size, seq_len_v, d_model)
        
        q = self.norm1(self.Wq(q))  # (batch_size, seq_len_q, d_model)
        k = self.norm2(self.Wk(k))  # (batch_size, seq_len_k, d_model)
        v = self.norm3(self.Wv(k))  # (batch_size, seq_len_v, d_model)
        # 分割为多个头
        
        q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)
        # print(q.shape, k.shape)
        # 计算缩放点积注意力
        if method == "dot":
            attention, attention_weights= self.scaled_dot_product_attention(q, k, v, mask)
        elif method == "kl":
            attention, attention_weights= self.ComputeKullbackLeiblerDivergence(q, k, v, mask)
        else:
            print("Please choose a calculation method!")       
        # scaled_attention: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        # 合并多个头
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(attention, (-1, attention.shape[1], self.d_model))  # (batch_size, seq_len_q, d_model)
        
        # 最终线性变换
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class InteractiveConvolutionBlock(tf.keras.Model):
    def __init__(self, filters=4):
        super(InteractiveConvolutionBlock, self).__init__()
        self.Conv1 = tf.keras.layers.Conv1D(filters, 2,activation="gelu",  padding='same') #activation="gelu",
        self.Conv2 = tf.keras.layers.Conv1D(filters, 2, activation="gelu", padding='same') #activation="gelu",
        self.Conv3 = tf.keras.layers.Conv1D(filters, 2, padding='same')
        self.Conv4 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1#, padding=None,
                                           # activation=activation
                                           )
        self.drop = tf.keras.layers.Dropout(rate=0.1)
        
    def call(self, x):
        
        if x.shape[1] == args.seq_len:
            targetship, aroundship = x[:, :, 0:args.featuer_num], x[:, :, args.featuer_num:]
        else:
            targetship, aroundship = x[:, 0:args.featuer_num, :], x[:, args.featuer_num:, :]
        # targetship, aroundship = x[:, :, 0:4], x[:, :, 4:]
        # target ship information
        x1 = self.Conv1(targetship)
        x1_1 = self.drop(x1)
        
        around_num = int(aroundship.shape[-1] / args.featuer_num)
        for i in range(around_num):
            
            if x.shape[1] == args.seq_len:
                xx = x[:, :, args.featuer_num*i:(i+1)*args.featuer_num]
            else:
                xx = x[:, args.featuer_num*i:(i+1)*args.featuer_num,  :]
            
            # around ship information
            x2 = self.Conv2(xx)
            x2_1 = self.drop(x2)

            # interactive information
            out1 = x1 * x2_1
            out2 = x2 * x1_1
            target_around = self.Conv3(out1 + out2)
            if i == 0:
                around_feature = tf.expand_dims(target_around, axis=3)
            else:
                around_feature = tf.concat([around_feature, tf.expand_dims(target_around, axis=3)]
                                           , axis=-1)
                
        #interactive information
        inter_infor = self.Conv4(around_feature)
        
        return tf.reshape(inter_infor, shape=(-1, inter_infor.shape[1], inter_infor.shape[2]))

class EmbeddingLayers(tf.keras.layers.Layer):
    def __init__(self,d_model, d_ff, dropout):
        super(EmbeddingLayers, self).__init__()
        
        self.dense = tf.keras.layers.Dense(d_ff, activation='gelu')
        self.dense_ = tf.keras.layers.Dense(d_model) 
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.dense(x)
        x = self.dropout(self.dense_(x))
        return self.ln(x)

class iTransBlock(tf.keras.Model):
    def __init__(self, layersnum, num_heads, d_model, d_ff, dropout):
        super(iTransBlock, self).__init__()
        self.layersnum = layersnum
        self.attention = MultiHeadAttention(hid_dim=d_ff, n_heads=num_heads)

        self.fcn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='gelu'), 
            tf.keras.layers.Dense(d_model), 
            tf.keras.layers.Dropout(dropout),
        ])
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        inputs = x
        for _ in range(self.layersnum):
            attn_out, atti = self.attention(inputs, inputs, method="kl")
            attn_out = self.norm1(inputs + attn_out)
            mlp_out = self.fcn(attn_out)
            inputs = self.norm2(attn_out + mlp_out)
        return inputs

class TransBlock(tf.keras.Model):
    def __init__(self, layersnum, num_heads, d_model, d_ff, dropout):
        super(TransBlock, self).__init__()
        self.layersnum = layersnum
        self.attention = MultiHeadAttention(hid_dim=d_ff, n_heads=num_heads)
        
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='gelu'),
            tf.keras.layers.Dense(d_model), 
            tf.keras.layers.Dropout(dropout),
        ])
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        inputs = x
        for _ in range(self.layersnum):
            attn_out, att = self.attention(inputs, inputs, method="kl")
            attn_out = self.norm1(inputs + attn_out)
            mlp_out = self.mlp(attn_out)
            inputs = self.norm2(attn_out + mlp_out)
        return inputs


class TransEncoder(tf.keras.Model):#tf.keras.Model
    def __init__(self, layersnum, num_heads, d_model, d_ff, dropout):
        super(TransEncoder, self).__init__()

        self.TICN = InteractiveConvolutionBlock(filters=128) 
        self.TICN1 = InteractiveConvolutionBlock(filters=128)
        
        self.embedding = EmbeddingLayers(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.embedding_1 = EmbeddingLayers(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        self.itrm_block = iTransBlock(layersnum=layersnum, num_heads=num_heads
                                  , d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        self.trm_block = TransBlock(layersnum=layersnum, num_heads=num_heads
                                  , d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.projection = tf.keras.layers.Dense(d_model, use_bias=True) 
        self.iprojection = tf.keras.layers.Dense(d_model, use_bias=True)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        # time-stream
        x_ = self.TICN(x)
        embedding_x1 = self.dropout(self.embedding(x_))
        trm_block_x1 = self.trm_block(embedding_x1)
        projection_x1 = self.projection(trm_block_x1)
          
        # feature-stream
        x = tf.transpose(x, perm=[0, 2, 1])
        x_ = self.TICN1(x)
        embedding_x = self.dropout(self.embedding_1(x_))
        trm_block_x = self.itrm_block(embedding_x)
        projection_x = self.iprojection(trm_block_x)
        
        # concat dual-stream parallel encoder output information
        projection = tf.concat([projection_x1, projection_x], axis=1)
  
        return projection


class CascadeTransDecoder(tf.keras.Model):
    def __init__(self, pred_len, num_heads, d_model, d_ff, dropout):
        super(CascadeTransDecoder, self).__init__()
        self.embedding = EmbeddingLayers(d_model=d_model, d_ff=d_ff, dropout=dropout) 
        self.seqlen = pred_len
        self.selfattention = MultiHeadAttention(hid_dim=d_ff, n_heads=num_heads) 
        self.crossattention = MultiHeadAttention(hid_dim=d_ff, n_heads=num_heads)
        self.projection = tf.keras.layers.Dense(2, use_bias=True)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, enc_values):      
        attention_weights = {}
        pred_i_vector = 1
        
        for i in range(self.seqlen):
            if i == 0:
                embedding_x = self.embedding(x)     
            else:
                x = tf.concat([x[:,1:,:], pred_i_vector], axis=1)
                embedding_x = self.embedding(x)
            self_att_x, attself = self.selfattention(embedding_x, embedding_x, method='kl')
            attn_out = self.norm1(embedding_x + self_att_x)
            cross_att_x, attcross = self.crossattention(attn_out, enc_values, method='kl')
            attn_out_ = self.norm2(attn_out + cross_att_x) 
            projection_x = self.projection(attn_out_)
            
            pred_i_vector = tf.expand_dims(projection_x[:, i, :], axis=1)
            
            if i == 0:
                output = pred_i_vector
            else: 
                output = tf.keras.layers.concatenate([output, pred_i_vector], axis=1)
        
            attention_weights['decoder_layer{}_attself'.format(i+1)] = attself
            attention_weights['decoder_layer{}_attcross'.format(i+1)] = attcross
        
        return output, attention_weights

# Hyperparameters
parser = argparse.ArgumentParser(description='Tensorflow KLDA-TICNNet Training')
# 实验数据参数
parser.add_argument('--seq_len', default=10, help="Input sequence length")
parser.add_argument('--featuer_num', default=4, help='Input featuer_num')
parser.add_argument('--pred_len', default=10, help='Output sequence length')
parser.add_argument('--out_num', default=2, help='Output dim==[lon, lat]')

# 模型结构参数
parser.add_argument('--layersnum', default=5, help='Transfromer layer number')
parser.add_argument('--d_model', default=192, help='Multi-head attention mapping dimension')
parser.add_argument('--num_heads', default=3, help='Multi-head attention head nums')
parser.add_argument('--d_ff', default=192, help='Fully connected network mapping dimension')

# 模型训练参数
parser.add_argument('--dropout', default=0.1, help="Dropout rate")
parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--epochs', type=int, default=3000, help="epochs") # 400
parser.add_argument('--bs', default=64, help="Batch_size")

args = parser.parse_args(args=[])

#--------------------------------------#
#                 实验数据获取           #
#-------------------------------------#
use_col = ['UnixTime_FEN', 'MMSI_', 'Course', 'Speed', 'Lon_d', 'Lat_d', 
        'df0_Course', 'df0_Speed', 'df0_Lon_d', 'df0_Lat_d',#'df0_MMSI_', 
        'df1_Course', 'df1_Speed', 'df1_Lon_d','df1_Lat_d',# 'df1_MMSI_', 
        # 'df2_Course', 'df2_Speed','df2_Lon_d', 'df2_Lat_d',# 'df2_MMSI_', 
        # 'df3_Course','df3_Speed', 'df3_Lon_d', 'df3_Lat_d',# 'df3_MMSI_'
        ] 


aisdata = pd.read_csv('daban.csv', usecols=use_col)

aisdata = aisdata.loc[:, use_col]
aisdata.fillna(value=0, inplace=True)

# 标准化最大最小值
lat_min, lat_max, lon_min, lon_max = 33.72, 34.18, 134.59, 135.20
cou_min, cou_max, spd_min, spd_max = 0., 359.9, 0., 17.6



for i in range(int((aisdata.shape[1] -2)/4)):
    aisdata.loc[:,aisdata.columns[int(2+4*i):int(6+4*i)]] = min_max_normalize(aisdata.loc[:,aisdata.columns[int(2+4*i):int(6+4*i)]].values
                                                          , lat_min, lat_max, lon_min, lon_max
                                                          , cou_min, cou_max, spd_min, spd_max)

timefun_ = lambda x: datetime.datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S")
timefun = lambda x: datetime.datetime.strptime(x[:-10],"%Y-%m-%dT%H:%M:%S")

aisdata['UnixTime_FEN'] = aisdata ['UnixTime_FEN'].apply(timefun_)
intfun = lambda x: int(x)
aisdata['MMSI_'] = aisdata['MMSI_'].apply(intfun)
aisdata = aisdata.loc[:, use_col]

aisdata.fillna(value=0, inplace=True)

def GenerateSamples(data, input_l, output_l):
    mmsi = list(np.unique(data.MMSI_))
    s_i, s_o = [], [] 
    for i in mmsi:
        data_mmsi = data[data['MMSI_'] == i]
        Samples_num = data_mmsi.shape[0] - (input_l + output_l) + 1
        for j in range(Samples_num):
            samples_i = data_mmsi.iloc[j:(j+input_l), 2:].values
            samples_o = data_mmsi.iloc[(input_l+j):(input_l+output_l+j), 4:6].values
            s_i.append(samples_i)
            s_o.append(samples_o)
  
    return np.array(s_i), np.array(s_o)

s_i, s_o = GenerateSamples(aisdata, input_l=args.seq_len, output_l=args.pred_len)

indecies = np.arange(s_i.shape[0])
# 打乱索引顺序
np.random.shuffle(indecies)

s_i, s_o = s_i[indecies], s_o[indecies]

train_enc_inputs = s_i[:int(s_i.shape[0]*0.8), :, :]                     
test_enc_inputs = s_i[int(s_i.shape[0]*0.8):, :, :]

train_dec_inputs, test_dec_inputs = train_enc_inputs[:, :, 2:4], test_enc_inputs[:, :, 2:4]

train_y = s_o[:int(s_o.shape[0]*0.8), :]
test_y = s_o[int(s_o.shape[0]*0.8):, :]

# Input Layer
enc_input = tf.keras.layers.Input(shape=([args.seq_len, s_i.shape[-1]]), name='enc_input')
dec_input = tf.keras.layers.Input(shape=([args.pred_len, 2]), name='dec_input')    

# 学习率衰减
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=args.lr, decay_steps= int(train_y.shape[0] // args.bs) * int(500) + 1
                    , decay_rate=0.93)

start_time = time.time()  
Loss, Mae, Val_Loss, Val_Mae =  pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
Ade, val_Ade, Fde, val_Fde = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
per_values, att_wh = pd.DataFrame(indecies), pd.DataFrame()
for i in range(20):
    # Encoder Layer
    enc_output = TransEncoder(layersnum=args.layersnum, num_heads=args.num_heads, d_model=args.d_model
                              , d_ff=args.d_ff, dropout=args.dropout)(enc_input)

    # Decoder Layer
    ouptut, att_weight = CascadeTransDecoder(pred_len=args.pred_len, num_heads=args.num_heads, d_model=args.d_model
                              , d_ff=args.d_ff, dropout=args.dropout)(dec_input, enc_output)
    DSATNet_model = tf.keras.models.Model(inputs=[enc_input, dec_input], outputs=[ouptut])
    DSATNet_model.compile(loss = tf.keras.losses.MeanSquaredError()
                ,optimizer = tf.keras.optimizers.Adam(learning_rate=exponential_decay)
              ,metrics=[tf.keras.metrics.MeanAbsoluteError(), HaversineDistanceMetric(), HaversineDistanceMetricFianl()])#, EuclideanDistanceMetric()

    history = DSATNet_model.fit([train_enc_inputs, train_dec_inputs], [train_y]
                      , epochs = args.epochs 
                      , batch_size = args.bs
                      , validation_data=([test_enc_inputs, test_dec_inputs], [test_y])
                      )
    # 获取注意力和结果
    attlayer_model = tf.keras.models.Model(inputs=DSATNet_model.input
                                    , outputs=DSATNet_model.layers[3].output)
    att_gam = attlayer_model.predict(x=[s_i, s_i[:,:, 2:4]])
    
    if i == 0:
        df_result = convert_all_to_dataframe(att_gam[1])
    else:
        df_result = pd.concat([df_result, convert_all_to_dataframe(att_gam[1])], axis=0)
    
    per_values.loc[:, [str(j + i) for j in range(20*(i), 20*(i+1))]] = tf.concat([att_gam[0][:,:,0], att_gam[0][:,:,1]], axis=-1)

per_values.to_csv('per_values.csv')
df_result.to_csv('att_values.csv')


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.xlabel('Epochs', fontsize = 12)
plt.ylabel('LOSS', fontsize = 12)
plt.show()
plt.plot(history.history['mean_absolute_error'], label='train')
plt.plot(history.history['val_mean_absolute_error'], label='test')
plt.legend()
plt.xlabel('Epochs', fontsize = 12)
plt.ylabel('MeanAbsoluteError', fontsize = 12)
plt.show()
