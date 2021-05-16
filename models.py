#!/usr/bin/env python  
#-*- coding:utf-8 -*- 

from layers import *


class SAHLSTM(object):
    def __init__(self,
                 granularity_num,
                 maxlen,
                 max_behaviors,
                 embedding_dims,
                 units,
                 head_num,
                 ff_units,
                 class_num=1,
                 last_activation='sigmoid'):
        self.granularity_num = granularity_num
        self.maxlen = maxlen
        self.max_behaviors = max_behaviors
        self.embedding_dims = embedding_dims
        self.units = units
        self.head_num = head_num
        self.ff_units = ff_units
        self.class_num = class_num
        self.last_activation = last_activation
        self.hrnn_ouputs = []

    def sequence_encoding(self, X, S, granularity_index):
        x_input = Lambda(lambda x: x[..., granularity_index])(X)
        s_input = mask = Lambda(lambda s: s[..., granularity_index])(S)
        x = Embedding(self.max_behaviors[granularity_index] + 1, self.embedding_dims, input_length=self.maxlen)(x_input)
        if granularity_index > 0:
            x = Concatenate(axis=-1)([x, self.hrnn_ouputs[granularity_index - 1]])
        s = Reshape((self.maxlen, 1))(s_input)
        inputs = Concatenate(axis=-1)([x, s])
        hrnn_output = HRNN(self.units, return_sequences=True)(inputs)
        self.hrnn_ouputs.append(hrnn_output)
        seq_att_output = SequenceAttention(head_num=self.head_num, ff_units=self.ff_units)(hrnn_output, mask=mask)
        output = Lambda(lambda x: K.expand_dims(x, 1))(seq_att_output)
        return output

    def get_model(self):
        X = Input((self.maxlen, self.granularity_num))
        S = Input((self.maxlen, self.granularity_num))

        sequence_outputs = []
        for i in range(self.granularity_num):
            sequence_output = self.sequence_encoding(X, S, i)
            sequence_outputs.append(sequence_output)

        x = Concatenate(axis=1)(sequence_outputs)
        x = StructureAttention(self.granularity_num)(x)

        output = Dense(1, activation=self.last_activation)(x)
        model = Model(input=[X, S], output=output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model


if __name__ == '__main__':
    model = SAHLSTM(granularity_num=3,
                    maxlen=500,
                    max_behaviors=[300, 200, 100],
                    embedding_dims=32,
                    units=32,
                    head_num=4,
                    ff_units=32).get_model()
    print(model.summary())