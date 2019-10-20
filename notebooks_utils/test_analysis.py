from analysis import count_recursive, find_path_to_node


def test_count_recursive():
    tree = {'LabelName': '1',
            'Subcategory': [
                {'LabelName': '2'},
                {'LabelName': '3',
                 'Subcategory': [
                     {'LabelName': '4'}
                 ]},
                {'LabelName': '5',
                 'Subcategory': [
                     {'LabelName': '6',
                      'Subcategory': [
                          {'LabelName': '7'}
                      ]}
                 ]},
                {'LabelName': '8'}
            ]}
    assert count_recursive(tree)[0] == 8
    assert count_recursive(tree)[1] == ['1', '2', '3', '4', '5', '6', '7', '8']


def test_find_path_to_node():
    tree = {'LabelName': '11',
            'Subcategory': [
                {'LabelName': '22'},
                {'LabelName': '33',
                 'Subcategory': [
                     {'LabelName': '44'}
                 ]},
                {'LabelName': '55',
                 'Subcategory': [
                     {'LabelName': '66',
                      'Subcategory': [
                          {'LabelName': '77'}
                      ]},
                     {'LabelName': '99',
                      'Subcategory': [
                          {'LabelName': '77'}
                      ]}
                 ]},
                {'LabelName': '88',
                 'Subcategory': [
                     {'LabelName': '77'}
                 ]}
            ]}
    assert find_path_to_node(tree, '77') == [('11', '55', '66', '77'),
                                             ('11', '55', '99', '77'),
                                             ('11', '88', '77')]
