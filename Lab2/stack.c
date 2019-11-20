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
#endif

	// The stack is always fine
	return 1;
}

int /* Return the type you prefer */
stack_push(int value, pthread_mutex_t stack_lock, stack_t** stack /* Make your own signature */)
{
#if NON_BLOCKING == 0
  // Implement a lock_based stack
	if(stack == NULL){
		return;
	}
	printf("pushing value %d\n", value);
	stack_t* element = malloc(sizeof(stack_t));
	element->change_this_member = value;
	pthread_mutex_lock(&stack_lock);
	(*stack)->next = element;
	element->prev = *stack;
	element->next = NULL;
	*stack = element;
	printf("stack head value = %d\n", (*stack)->change_this_member);
	printf("element value = %d\n", element->change_this_member);
	pthread_mutex_unlock(&stack_lock);
	return 0;
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*)1);

  return 0;
}

int /* Return the type you prefer */
stack_pop(pthread_mutex_t stack_lock, stack_t** stack/* Make your own signature */)
{
#if NON_BLOCKING == 0
	pthread_mutex_lock(&stack_lock);
	int value = (*stack)->change_this_member;
	if((*stack)->prev == NULL){
		printf("prev =  null\n");
	}
	stack_t* prev_element = (*stack)->prev;
	prev_element->next = NULL;
	free(*stack);
	*stack=prev_element;
	pthread_mutex_unlock(&stack_lock);
	return value;
  // Implement a lock_based stack
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return 0;
}
