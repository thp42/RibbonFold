### Install treelib: pip install treelib

import treelib
from treelib import Node, Tree
import os, sys

############################
### List functions
############################

def list_flatten(instance, parent_node):
    """
    Flatten the list object
    """
    assert isinstance(instance, list)
    tree = parent_node.data['tree']
    
    array_data = []
    node = tree.create_node(tag=list, parent=parent_node.identifier, data={'tree': tree})
    for item in instance:
        sub_tree_array_data, sub_tree_root = _genetic_flatten(item, node)
        array_data += sub_tree_array_data
        sub_tree_root.set_predecessor(node.identifier, tree.identifier)
    
    return array_data, node

def list_unflatten(node, array_data):
    """
    Convert the flattend obj to list
    """
    assert isinstance(node, treelib.node.Node)
    assert node.tag == list
    assert isinstance(array_data, list)
    tree = node.data['tree']
    
    nested_data = []
    consumed = 0
    for node_id in node.successors(tree.identifier):
        son_node = tree.get_node(node_id)
        sub_tree_nested_data, idx = _genetic_unflatten(son_node, array_data)
        nested_data.append( sub_tree_nested_data )
        array_data = array_data[idx:]
        consumed += idx
    
    return nested_data, consumed

############################
### Tuple functions
############################

def tuple_flatten(instance, parent_node):
    """
    Flatten the tuple object
    
    Tuple can be namedtuple !!!!
    """
    assert isinstance(instance, tuple)
    tree = parent_node.data['tree']
    
    array_data = []
    node = tree.create_node(tag=tuple, parent=parent_node.identifier, data={'tree': tree, 'type': type(instance)})
    for item in instance:
        sub_tree_array_data, sub_tree_root = _genetic_flatten(item, node)
        array_data += sub_tree_array_data
        sub_tree_root.set_predecessor(node.identifier, tree.identifier)
    
    return array_data, node

def tuple_unflatten(node, array_data):
    """
    Convert the flattend obj to list
    """
    assert isinstance(node, treelib.node.Node)
    assert node.tag == tuple
    assert isinstance(array_data, list)
    tree = node.data['tree']
    dtype = node.data['type']
    
    nested_data = []
    consumed = 0
    for node_id in node.successors(tree.identifier):
        son_node = tree.get_node(node_id)
        sub_tree_nested_data, idx = _genetic_unflatten(son_node, array_data)
        nested_data.append( sub_tree_nested_data )
        array_data = array_data[idx:]
        consumed += idx
    
    try:
        ### is a namedtuple
        nested_data = dtype(*nested_data)
    except TypeError:
        nested_data = dtype(nested_data)
    
    return nested_data, consumed

############################
### Dict functions
############################

def dict_flatten(instance, parent_node):
    """
    Flatten the dict object
    """
    assert isinstance(instance, dict)
    tree = parent_node.data['tree']
    
    key_array = []
    array_data = []
    node = tree.create_node(tag=dict, parent=parent_node.identifier, data={'tree': tree})
    for key, item in instance.items():
        key_array.append(key)
        sub_tree_array_data, sub_tree_root = _genetic_flatten(item, node)
        array_data += sub_tree_array_data
        sub_tree_root.set_predecessor(node.identifier, tree.identifier)
    
    node.data['keys'] = key_array
    return array_data, node

def dict_unflatten(node, array_data):
    """
    Convert the flattend obj to dict
    """
    assert isinstance(node, treelib.node.Node)
    assert node.tag == dict
    assert isinstance(array_data, list)
    tree = node.data['tree']
    
    nested_data = []
    consumed = 0
    for node_id in node.successors(tree.identifier):
        son_node = tree.get_node(node_id)
        sub_tree_nested_data, idx = _genetic_unflatten(son_node, array_data)
        nested_data.append( sub_tree_nested_data )
        array_data = array_data[idx:]
        consumed += idx
    
    nested_data = dict(zip(node.data['keys'], nested_data))
    return nested_data, consumed

############################
### Relation ships
############################

DT_FLATTEN = {
    list: list_flatten,
    tuple: tuple_flatten,
    dict: dict_flatten
}

DT_UNFLATTEN = {
    list: list_unflatten,
    tuple: tuple_unflatten,
    dict: dict_unflatten
}

def _genetic_flatten(instance, node):
    """
    Flatten the nested object
    
    Parameters
    --------------
    instance: obj
        list, tuple, dict, ...
    
    node: treelib.node.Node
        The root or non-leaf node
    
    Return
    --------------
    array_data: flatten data
    node: treelib.node.Node
    """
    assert isinstance(node, treelib.node.Node)
    tree = node.data['tree']
    
    for dt_type in DT_FLATTEN:
        if isinstance(instance, dt_type):
            array_data, node = DT_FLATTEN[dt_type](instance, node)
            return array_data, node
    leaf = tree.create_node(tag="leaf", parent=node.identifier, data={'tree': tree})
    return [instance], leaf

def _genetic_unflatten(node, array_data):
    """
    Convert the flatten object to nested object
    
    Parameters
    --------------
    node: treelib.node.Node
        The root or non-leaf node
    
    array_data: flatten data
    
    Return
    --------------
    nested_data: nested data
    consumed: int
        The count of data to construct the nested_data from array_data
    """
    assert isinstance(node, treelib.node.Node)
    tree = node.data['tree']
    
    for dt_type in DT_UNFLATTEN:
        if node.tag == dt_type:
            nested_data, consumed = DT_UNFLATTEN[dt_type](node, array_data)
            return nested_data, consumed
    if node.tag == 'leaf':
        return array_data[0], 1
    raise RuntimeError(f"Unexpected node type: {node.tag}")


############################
### User functions
############################

def tree_flatten(instance):
    """
    Flatten the nested object
    
    Parameters
    --------------
    instance: obj
        list, tuple, dict, ...
    
    Return
    --------------
    array_data: flatten data
    
    tree_def: treelib.node.Node
        Define the relationship, to reconstruct the nested data
    """
    tree = Tree()
    root_node = tree.create_node(tag='root', data={'tree': tree})
    array_data, tree_def = _genetic_flatten(instance, root_node)
    return array_data, tree_def

def tree_unflatten(tree_def, array_data):
    """
    Convert the flatten object to nested object
    
    Parameters
    --------------
    tree_def: treelib.node.Node
        Define the relationship, to reconstruct the nested data
    
    array_data: flatten data
    
    Return
    --------------
    nested_data: nested data
    """
    nested_data, consumed = _genetic_unflatten(tree_def, array_data)
    if consumed != len(array_data):
        print(f"Warning: Expect consumed {len(array_data)} variables, but got {consumed}", file=sys.stderr)
    return nested_data


def register_pytree_node(nodetype, flatten_func, unflatten_func):
    """
    Add a new nested datatype to system.
    To defined a new nested datatype, the following points should be noted:
    
    1. flatten_func(nested_data, tree_node_type) -> array_data, new_tree_node_type:
        1.1 Create a new node in the flatten_func, the tag should be set as data type
            node = tree.create_node(tag=dict, parent=parent_node.identifier, data={'tree': tree})
        1.2 The new generated node should be set predecessor:
            sub_tree_root.set_predecessor(node.identifier, tree.identifier)
    2. unflatten_func(tree_node_type, array_data) -> nested_data, consumed
        1.1 A int type consumed should record how many data are consumed
        1.2 The nested data should be convert to corresponding data type
            nested_data = dict(zip(node.data['keys'], nested_data))
    """
    global DT_FLATTEN
    global DT_UNFLATTEN

    if nodetype in DT_FLATTEN:
        print(f"Warning: {nodetype} has been registered. Re-register again.", file=sys.stderr)
    
    DT_FLATTEN[nodetype] = flatten_func
    DT_UNFLATTEN[nodetype] = unflatten_func


def tree_map(apply_fn, nested_data, *rest):
    """
    Apply the function to each element of the nested data structure.
    
    Parameters
    -----------
    apply_fn: function
        The number of parameters should equal to the number of parameters: 1 + len(rest)
    
    nested_data: nested data structure
    
    Return
    ----------
    nested_data: Processed data structure
    
    Example
    ----------
    nested_data = [ {'a': [1,2,3]}, [4,5,[6,7,8]], (2,3,4), [9,{"b": 222}] ]
    new_data = tree_multimap(lambda x, y: x + y, nested_data, 1)
    new_data
    """
    #print("tree_map called")
    array_data, tree_def = tree_flatten(nested_data)
    for idx in range(len(array_data)):
        array_data[idx] = apply_fn( array_data[idx], *rest )
    nested_data = tree_unflatten(tree_def, array_data)
    return nested_data

def tree_multimap(apply_fn, nested_data, *rest_nested_data):
    """
    Apply the function to each element of the nested data structure. Multiple nested_data can be provided, but their 
    shape should be same.
    
    Parameters
    -----------
    apply_fn: function
        The number of parameters should equal to the number of parameters: 1 + len(rest)
    
    nested_data: nested data structure
    
    rest_nested_data: other nested data structure
    
    Return
    ----------
    nested_data: Processed data structure
    
    Example
    ----------
    nested_data = [ {'a': [1,2,3]}, [4,5,[6,7,8]], (2,3,4), [9,{"b": 222}] ]
    new_data = tree_multimap(lambda x, y, z: x+y+z, nested_data, nested_data, nested_data)
    new_data
    """
    array_data, tree_def = tree_flatten(nested_data)
    rest_data = []
    #print(len(rest_nested_data))
    for rest_node in rest_nested_data:
        rest_array_data, _ = tree_flatten(rest_node)
        assert len(rest_array_data) == len(array_data), f"Error: Expect same length, but got {len(rest_array_data)}, {len(array_data)}"
        rest_data.append(rest_array_data)
    
    for idx in range(len(array_data)):
        rest = [ d[idx] for d in rest_data ]
        array_data[idx] = apply_fn( array_data[idx], *rest )
    
    nested_data = tree_unflatten(tree_def, array_data)
    return nested_data

