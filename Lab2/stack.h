/*
 * stack.h
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
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H

typedef struct stack stack_t;
struct stack
{
	stack_t* next;
  int change_this_member;
};
typedef struct stack stack_t;

struct aba_thread_args
{
  int id;
  pthread_barrier_t *barrier1;
  pthread_barrier_t *barrier2;
  pthread_barrier_t *barrier3;
  pthread_mutex_t *stack_lock;
  stack_t **stack;
  stack_t **free_list;
};
typedef struct aba_thread_args aba_thread_args_t;

int stack_push(int, pthread_mutex_t, stack_t**, stack_t** /* Make your own signature */);
int stack_pop(pthread_mutex_t, stack_t**, stack_t** /* Make your own signature */);

void *thread_0_stack_pop(void *arg);
void *thread_1_stack_pop(void *arg);
void *thread_2_stack_pop(void *arg);

/* Use this to check if your stack is in a consistent state from time to time */
int stack_check(stack_t *stack);
#endif /* STACK_H */
