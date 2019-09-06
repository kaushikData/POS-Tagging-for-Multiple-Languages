import sys
import tensorflow as tf
import numpy
from Read_Data import DatasetReader
from Sequential_Model import SequenceModel

def main():
    
    reader = DatasetReader
    train_filename = sys.argv[1]
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data

    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    for j in range(10):
        model.train_epoch(train_terms, train_tags, train_lengths)
        print('Finished epoch %i. Evaluating ...' % (j + 1))
        model.evaluate(test_terms, test_tags, test_lengths)


if __name__ == '__main__':
    main()