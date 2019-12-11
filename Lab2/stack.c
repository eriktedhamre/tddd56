/*
 * stack.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 *
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass
	assert(1 == 1);

	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
	stack_t* current_element = stack;
	printf("in stackcheck \n");
	while(1){
		assert(current_element != NULL);
		printf("stack = %d, next = %d, memory = %d,value = %d\n", stack, current_element->next, current_element, current_element->change_this_member);
		if(current_element->next != NULL){
			current_element = current_element->next;
		} else{
			break;
		}
	}
#endif

	// The stack is always fine
	return 1;
}

int /* Return the type you prefer */
stack_push(int value, pthread_mutex_t *stack_lock, stack_t** stack, stack_t** free_list /* Make your own signature */)
{
	if(*stack == *free_list){
		printf("STACK == FREELIST");
	}
#if NON_BLOCKING == 0
  // Implement a lock_based stack

	stack_t *element;
	pthread_mutex_lock(stack_lock);
	if(*free_list == NULL){
#if MEASURE == 0 || MEASURE == 1
		printf("free_list = null\n");
#endif
		element = malloc(sizeof(stack_t));
#if MEASURE == 0 || MEASURE == 1
		printf("malloc memory = %d\n", element);
#endif
	} else {
#if MEASURE == 0 || MEASURE == 1
		printf("*free list != null = %d\n", *free_list);
#endif
		element = *free_list;
		if(element->next != NULL){
			*free_list = element->next;
		} else{
			*free_list = NULL;
		}
	}
	element->change_this_member = value;
#if MEASURE == 0 || MEASURE == 1
	printf("pushing value %d\n", value);
#endif
	element->next = *stack;
	*stack = element;
#if MEASURE == 0 || MEASURE == 1
	printf("unlocking push lock\n");
#endif
	pthread_mutex_unlock(stack_lock);
#if MEASURE == 0 || MEASURE == 1
	printf("stack head value = %d\n", (*stack)->change_this_member);
#endif
	return 0;



#elif NON_BLOCKING == 1
	stack_t* element;
	if(*free_list == NULL){
	#if MEASURE == 0 || MEASURE == 1
		printf("free_list = null\n");
	#endif
		element = malloc(sizeof(stack_t));
	} else {
	#if MEASURE == 0 || MEASURE == 1
		printf("free_list != null\n");
	#endif
		stack_t * new;
		do{
			element = *free_list;
			new = element->next;
			new = NULL;
		#if MEASURE == 0 || MEASURE == 1
			printf("element = %d, element->next=%d, new = %d\n", element, element->next, new);
		#endif
		} while(cas(free_list, element, new) != element);
/*
		element = *free_list;
		if(element->next != null){
			*free_list = element->next;
		} else{
			*free_list = NULL
		}
*/
	}
	stack_t* old;
	element->change_this_member = value;
#if MEASURE == 0 || MEASURE == 1
	printf("pushing value %d\n", value);
#endif
	do{
		old = *stack;
		element->next = old;
	} while(cas(stack, old, element) != old);
#if MEASURE == 0 || MEASURE == 1
	printf("stack head value = %d\n", (*stack)->change_this_member);
	printf("element value = %d\n", element->change_this_member);
#endif
  // Implement a harware CAS-based stack
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses

	stack_check((stack_t*) *stack);

  return 0;
}

int /* Return the type you prefer */
stack_pop(pthread_mutex_t *stack_lock, stack_t** stack,stack_t** free_list/* Make your own signature */)
{
	if(*stack == *free_list){
		printf("STACK == FREELIST");
	}
#if NON_BLOCKING == 0
#if MEASURE == 0 || MEASURE == 1
	printf("memory = %d\n", *stack);
#endif
	pthread_mutex_lock(stack_lock);
#if MEASURE == 0 || MEASURE == 1
	printf("in lock memory = %d\n", *stack);
#endif
	int value = (*stack)->change_this_member;
	if((*stack)->next == NULL){
		printf("next =  null\n");
		printf("value = %d, stack->value = %d, free_list->value = %d, stack = %d, free_list = %d, stack->next = %d, free_list->next = %d\n", value, (*stack)->change_this_member, (*free_list)->change_this_member, *stack, *free_list, (*stack)->next, (*free_list)->next);
	}
	stack_t* prev_element = (*stack)->next;
	(*stack)->next = *free_list;
	(*stack)->change_this_member = -1;
	*free_list = *stack;
	*stack = prev_element;
	#if MEASURE == 0 || MEASURE == 1
	printf("value = %d, stack->value = %d, free_list->value = %d, stack = %d, free_list = %d, stack->next = %d, free_list->next = %d\n", value, (*stack)->change_this_member, (*free_list)->change_this_member, *stack, *free_list, (*stack)->next, (*free_list)->next);
#endif
	pthread_mutex_unlock(stack_lock);
	return value;
  // Implement a lock_based stack




#elif NON_BLOCKING == 1
stack_t* old;
stack_t* next_element;
do{
	old = *stack;
	next_element = old->next;
} while(cas(stack, old, next_element) != old);
	int value = old->change_this_member;
#if MEASURE == 1
	printf("value = %d\n", value);
	// Implement a harware CAS-based stack

	printf("pushing to free_list\n");
#endif
	stack_t* old_free_list;
	do{
		old_free_list = *free_list;
		old->next = old_free_list;
	} while(cas(free_list, old_free_list, old) != old_free_list);
	return value;

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return 0;
}

#if NON_BLOCKING ==1 || NON_BLOCKING == 2

void *thread_0_stack_pop(void *arg)
{
	aba_thread_args_t *args = (aba_thread_args_t*)arg;
	pthread_barrier_t *barrier1 = args->barrier1;
	pthread_barrier_t *barrier2 = args->barrier2;
	pthread_barrier_t *barrier3 = args->barrier3;
  pthread_mutex_t *stack_lock = args->stack_lock;
  stack_t **stack = (args->stack);
  stack_t **free_list = (args->free_list);

	stack_t* old;
	stack_t* next_element;
	do{
		old = *stack;
		next_element = old->next;
		printf("thread 0 barrier 1\n");
		pthread_barrier_wait(barrier1);
		printf("thread 0 barrier 2\n");
		pthread_barrier_wait(barrier2);
		printf("thread 0 barrier 3\n");
		pthread_barrier_wait(barrier3);
	} while(cas(stack, old, next_element) != old);
		int value = old->change_this_member;
		printf("value = %d\n", value);
	  // Implement a harware CAS-based stack

		printf("pushing to free_list\n");
		stack_t* old_free_list;
		do{
			old_free_list = *free_list;
			old->next = old_free_list;
		} while(cas(free_list, old_free_list, old) != old_free_list);

}

void *thread_1_stack_pop(void *arg)
{
	aba_thread_args_t *args = (aba_thread_args_t*)arg;
	pthread_barrier_t *barrier1 = args->barrier1;
	pthread_barrier_t *barrier2 = args->barrier2;
	pthread_barrier_t *barrier3 = args->barrier3;
  pthread_mutex_t *stack_lock = args->stack_lock;
  stack_t **stack = args->stack;
  stack_t **free_list = args->free_list;

	pthread_mutex_lock(stack_lock);
	printf("thread 1 barrier 1\n");
	pthread_barrier_wait(barrier1);
	stack_t* value_to_push;
	stack_t* old;
	stack_t* next_element;
	do{
		old = value_to_push = *stack;
		next_element = old->next;
	} while(cas(stack, old, next_element) != old);
		int value = old->change_this_member;
		printf("value = %d\n", value);
	  // Implement a harware CAS-based stack

		printf("pushing to free_list\n");
		stack_t* old_free_list;
		do{
			old_free_list = *free_list;
			old->next = old_free_list;
		} while(cas(free_list, old_free_list, old) != old_free_list);
	pthread_mutex_unlock(stack_lock);
	printf("thread 1 barrier 2\n");
	pthread_barrier_wait(barrier2);
	stack_t* element = value_to_push;
	printf("pushing value %d\n", value);
	do{
		old = *stack;
		element->next = old;
	} while(cas(stack, old, element) != old);
	printf("stack head value = %d\n", (*stack)->change_this_member);
	printf("element value = %d\n", element->change_this_member);
	printf("thread 1 barrier 3\n");
	pthread_barrier_wait(barrier3);

}

void *thread_2_stack_pop(void *arg)
{
	aba_thread_args_t *args = (aba_thread_args_t*)arg;
	pthread_barrier_t *barrier1 = args->barrier1;
	pthread_barrier_t *barrier2 = args->barrier2;
	pthread_barrier_t *barrier3 = args->barrier3;
  pthread_mutex_t *stack_lock = args->stack_lock;
  stack_t **stack = args->stack;
  stack_t **free_list = args->free_list;


	printf("thread 2 barrier 1\n");
	pthread_barrier_wait(barrier1);
	pthread_mutex_lock(stack_lock);
	stack_t* old;
	stack_t* next_element;
	do{
		old = *stack;
		next_element = old->next;
	} while(cas(stack, old, next_element) != old);
		int value = old->change_this_member;
		printf("value = %d\n", value);
	  // Implement a harware CAS-based stack

		printf("pushing to free_list\n");
		stack_t* old_free_list;
		do{
			old_free_list = *free_list;
			old->next = old_free_list;
		} while(cas(free_list, old_free_list, old) != old_free_list);
	pthread_mutex_unlock(stack_lock);
	printf("thread 2 barrier 2\n");
	pthread_barrier_wait(barrier2);
	printf("thread 2 barrier 3\n");
	pthread_barrier_wait(barrier3);
}

#endif
