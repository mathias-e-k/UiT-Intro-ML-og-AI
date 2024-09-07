# Imports
from __future__ import annotations  # Needed for typing Node class

import warnings
from typing import Any

import binarytree
import heapdict


# Node class (do not change)
class Node:
    def __init__(self, data: Any = None, next: None | Node = None):
        self.data = data
        self.next = next


# Add your implementations below


class Stack:
    def __init__(self):
        """Initialize stack object, with head attribute"""
        self.head = None

    def push(self, data: Any) -> None:
        """Add new node with data to stack"""
        self.head = Node(data, self.head)

    def peek(self) -> Any | None:
        """Return data from node on top of stack, without changing stack"""
        return self.head.data

    def pop(self) -> Any:
        """Remove last added node and return its data"""
        if self.head == None:
            raise IndexError("Stack is empty")
        node = self.head
        self.head = node.next
        return node.data


class Queue:
    def __init__(self):
        """Initialize queue object with head and tail"""
        self.head = None
        self.tail = None

    def enqueue(self, data: Any) -> None:
        """Add node with data to queue"""
        new_node = Node(data)
        if self.tail is not None:
            self.tail.next = new_node
        self.tail = new_node
        if self.head is None:
            self.head = self.tail

    def peek(self) -> Any | None:
        """Return data from head of queue without changing the queue"""
        return self.head.data

    def dequeue(self) -> Any:
        """Remove node from head of queue and return its data"""
        if self.head == None:
            raise IndexError("Queue is empty")
        node = self.head
        self.head = node.next
        if self.head is None:
            self.tail = None
        return node.data


class EmergencyRoomQueue:
    def __init__(self):
        """Initialize emergency room queue, use heapdict as property 'queue'"""
        self.queue = heapdict.heapdict()

    def add_patient_with_priority(self, patient_name: str, priority: int) -> None:
        """Add patient name and priority to queue

        # Arguments:
        patient_name:   String with patient name
        priority:       Integer. Higher priority corresponds to lower-value number.
        """
        self.queue[patient_name] = priority

    def update_patient_priority(self, patient_name: str, new_priority: int) -> None:
        """Update the priority of a patient which is already in the queue

        # Arguments:
        patient_name:   String, name of patient in queue
        new_priority:   Integer, updated priority for patient

        """
        self.queue[patient_name] = new_priority

    def get_next_patient(self) -> str:
        """Remove highest-priority patient from queue and return patient name

        # Returns:
        patient_name    String, name of patient with highest priority
        """
        name, priority = self.queue.popitem()
        return name


class BinarySearchTree:
    def __init__(self, root: binarytree.Node | None = None):
        """Initialize binary search tree

        # Inputs:
        root:    (optional) An instance of binarytree.Node which is the root of the tree

        # Notes:
        If a root is supplied, validate that the tree meets the requirements
        of a binary search tree (see property binarytree.Node.is_bst ). If not, raise
        ValueError.
        """
        if root is not None and not root.is_bst:
            raise ValueError("Root is not a valid BST")
        self.root = root

    def insert(self, value: float | int) -> None:
        """Insert a new node into the tree (binarytree.Node object)

        # Inputs:
        value:    Value of new node

        # Notes:
        The method should issue a warning if the value already exists in the tree.
        See https://docs.python.org/3/library/warnings.html#warnings.warn
        In the case of duplicate values, leave the tree unchanged.
        """
        new_node = binarytree.Node(value)
        if self.root is None:
            self.root = new_node
            return
        current_node = self.root
        done = False
        while not done:
            if new_node.value == current_node.value:
                warnings.warn("Duplicate value. Value already exists in tree")
                done = True

            elif new_node.value < current_node.value:
                if current_node.left is None:
                     current_node.left = new_node
                     done = True
                else:
                    current_node = current_node.left
            
            elif new_node.value > current_node.value:
                if current_node.right is None:
                     current_node.right = new_node
                     done = True
                else:
                    current_node = current_node.right

    def __str__(self) -> str | None:
        """Return string representation of tree (helper function for debugging)"""
        if self.root is not None:
            return str(self.root)
