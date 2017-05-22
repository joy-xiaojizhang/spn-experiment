import word2vec
import numpy as np
import parse_movie_set

def train():
    movie_set = cornell_movie_set.MovieSet()
    movie_set.parse_movie_set('train')
    word2vec.word2phrase('cornell_movie_train.txt', 'movie_phrases_train.txt', verbose=True)
    word2vec.word2vec('movie_phrases_train.txt', 'movie_train.bin', size=100, verbose=True)
    model = word2vec.load('movie_train.bin')
    return model

def test():
    movie_set = cornell_movie_set.MovieSet()
    movie_set.parse_movie_set('test')
    word2vec.word2phrase('cornell_movie_test.txt', 'movie_phrases_test.txt', verbose=True)
    word2vec.word2vec('movie_phrases_test.txt', 'movie_test.bin', size=100, verbose=True)
    model = word2vec.load('movie_test.bin')
    return model

def create_embedding(datatype='train'):
    if datatype == 'train':
        model = train()
        mat = []
        fo = open('cornell_movie_train.txt', 'r')
        for line in fo:
            line = line.rstrip('\n').split(' ')
            for word in line:
                try:
                    c = model[word]
                except:
                    mat.append([0] * 100)
                    continue
                mat.append(model[word].tolist())
        mat = np.array(mat)
        # Reshape to conform with input placeholder
        # (Avoid Tensorflow ValueError)
        mat = np.reshape(mat, (-1, 2000))
        fo.close()
        np.savetxt('movietrain.txt', mat, fmt='%.4f', delimiter=',')
    else:
        model = test()
        mat = []
        fo = open('cornell_movie_test.txt', 'r')
        for line in fo:
            line = line.rstrip('\n').split(' ')
            for word in line:
                try:
                    c = model[word]
                except:
                    mat.append([0] * 100)
                    continue
                mat.append(model[word].tolist())
        mat = np.array(mat)
        mat = np.reshape(mat, (-1, 1000))
        fo.close()
        np.savetxt('movietest.txt', mat, fmt='%.4f', delimiter=',')

if __name__ == '__main__':
    create_embedding('train')
    create_embedding('test')
