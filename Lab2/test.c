/*
 * test.c
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

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>
#include <semaphore.h>

#include "test.h"
#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

/* Helper function for measurement */
double timediff(struct timespec *begin, struct timespec *end)
{
	double sec = 0.0, nsec = 0.0;
   if ((end->tv_nsec - begin->tv_nsec) < 0)
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec  - 1);
      nsec = (double)(end->tv_nsec - begin->tv_nsec + 1000000000);
   } else
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec );
      nsec = (double)(end->tv_nsec - begin->tv_nsec);
   }
   return sec + nsec / 1E9;
}

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

#ifndef NDEBUG
int
assert_fun(int expr, const char *str, const char *file, const char* function, size_t line)
{
	if(!(expr))
	{
		fprintf(stderr, "[%s:%s:%zu][ERROR] Assertion failure: %s\n", file, function, line, str);
		abort();
		// If some hack disables abort above
		return 0;
	}
	else
		return 1;
}
#endif

//#if NON_BLOCKING == 0
pthread_mutex_t stack_lock;
//#endif

int queue;
stack_t *stack;
stack_t *free_list;
data_t data;

#if MEASURE != 0
struct stack_measure_arg
{
  int id;
  stack_t** stack;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1 || MEASURE == 3
void*
stack_measure_pop(void* arg)
  {
    stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
    int i;
    //stack_t* stack = *(args->stack);

    clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
    for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
      {
        stack_pop(stack_lock, &stack, &free_list);
        // See how fast your implementation can pop MAX_PUSH_POP elements in parallel
      }
    clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

    return NULL;
  }
#endif
#if MEASURE == 2 || MEASURE == 3
void*
stack_measure_push(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;
  //stack_t * stack = *(args->stack);

  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
    {
        stack_push(i, stack_lock, &stack, &free_list);
        // See how fast your implementation can push MAX_PUSH_POP elements in parallel
    }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#endif
#if MEASURE == 3
void* stack_measure_push_pop(void* arg){
  stack_measure_push(arg);
  stack_measure_pop(arg);

  return NULL;
}
#endif
#endif

/* A bunch of optional (but useful if implemented) unit tests for your stack */
void
test_init()
{
  // Initialize your test batch
}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;

  // Allocate a new stack and reset its values
  stack = malloc(sizeof(stack_t));
  free_list = malloc(sizeof(stack_t));

  // Reset explicitely all members to a well-known initial value
  // For instance (to be deleted as your stack design progresses):
  stack->change_this_member = 0;
  /*
  stack_push(1, stack_lock, &stack, &free_list);
  stack_push(2, stack_lock, &stack, &free_list);
  stack_push(3, stack_lock, &stack, &free_list);
  stack_push(4, stack_lock, &stack, &free_list);
  */
}

void
test_teardown()
{
  // Do not forget to free your stacks after each test
  // to avoid memory leaks
  /*
  stack_pop(stack_lock, &stack, &free_list);
  stack_pop(stack_lock, &stack, &free_list);
  stack_pop(stack_lock, &stack, &free_list);
  stack_pop(stack_lock, &stack, &free_list);
  stack_push(1, stack_lock, &stack, &free_list);
  stack_pop(stack_lock, &stack, &free_list);
  */
  free(stack);

}

void
test_finalize()
{
  // Destroy properly your test batch
}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it

  // Do some work
  printf("before stack_push\n");

  stack_push(1, stack_lock, &stack, &free_list /* add relevant arguments here */);
  printf("after stack_push\n");
  stack_pop(stack_lock, &stack, &free_list);
  printf("after stack_pop\n");

  // check if the stack is in a consistent state
  printf("before stack_check assert\n");
  int res = assert(stack_check(stack));
  printf("after stack_check assert\n");

  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  // Now, the test succeeds
  return res && assert(stack->change_this_member == 0);
}

int
test_pop_safe()
{
  // Same as the test above for parallel pop operation
  stack_push(1, stack_lock, &stack, &free_list /* add relevant arguments here */);
  int value = stack_pop(stack_lock, &stack, &free_list);
  int res = assert(stack_check(stack));
  printf("value = %d, res = %d, stack->member=%d\n", value, res, stack->change_this_member);
  // For now, this test always fails
  return res && assert(value == 1) && assert(stack->change_this_member == 0);
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

int
test_aba()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  int success, aba_detected = 0;
  // Write here a test for the ABA problem


  // Populate stack
  stack_push(2, stack_lock, &stack, &free_list);
  stack_push(3, stack_lock, &stack, &free_list);
  stack_push(4, stack_lock, &stack, &free_list);

  // Setup 2 barriers
  pthread_barrier_t barrier1;
  pthread_barrier_t barrier2;
  pthread_barrier_t barrier3;
  pthread_barrier_init(&barrier1, NULL, ABA_NB_THREADS);
  pthread_barrier_init(&barrier2, NULL, ABA_NB_THREADS);
  pthread_barrier_init(&barrier3, NULL, ABA_NB_THREADS);

  // Fix arguments for the threads
  pthread_t aba_threads[ABA_NB_THREADS];
  aba_thread_args_t arguments[ABA_NB_THREADS];
  for (int i = 0; i < ABA_NB_THREADS; i++) {
    arguments[i].id = i;
    arguments[i].barrier1 = &barrier1;
    arguments[i].barrier2 = &barrier2;
    arguments[i].barrier3 = &barrier3;
    arguments[i].stack_lock = &stack_lock;
    arguments[i].stack= &stack;
    arguments[i].free_list = &free_list;
  }
  //Create 3 Thread
  pthread_create(&aba_threads[0], NULL, thread_0_stack_pop, (void*)&arguments[0]);
  pthread_create(&aba_threads[1], NULL, thread_1_stack_pop, (void*)&arguments[1]);
  pthread_create(&aba_threads[2], NULL, thread_2_stack_pop, (void*)&arguments[2]);

  for (int t = 0; t < ABA_NB_THREADS; t++)
	{
		pthread_join(aba_threads[t], NULL);
	}
  //Barrier1 1 2
  //0 Start Pop
  //old_head == A, old_head == B
  //Lock 0
  //1 Pop A
  //2 Pop B
  //1 Push A
  //0 Resume Pop
  printf("stack value =%d\n", stack->change_this_member);
  success = aba_detected;
  stack_check(&stack);
  return success;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
#if NON_BLOCKING == 1
      printf("performed action number %d\n",i);
      } while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
      } while (software_cas(args->counter, old, local, args->lock) != old);
#endif
    }
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int
test_cas()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;
  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = assert(counter == (size_t)(NB_THREADS * MAX_PUSH_POP));

  if (!success)
    {
      printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
    }

  return success;
#else

  return 1;
#endif
}

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
queue = 0;
// MEASURE == 0 -> run unit tests
//#if NON_BLOCKING == 0
if(pthread_mutex_init(&stack_lock, NULL) != 0){
  printf("queue_lock init failed\n");
}
//#endif
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

  test_finalize();
#else
  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];
  pthread_attr_init(&attr);
  free_list = malloc(sizeof(stack_t));
  free_list->change_this_member = -2;
  for (i = 0; i <= MAX_PUSH_POP;i++){
    stack_push(i,stack_lock, &stack, &free_list);
  }
  for (i = 0; i < MAX_PUSH_POP; i++) {
    stack_t* free_list_element = malloc(sizeof(stack_t));
    free_list_element->next = free_list;
    free_list_element->change_this_member = -3;
    free_list = free_list_element;
  }

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
      //arg[i].stack = &stack;
#if MEASURE == 1
      pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#elif MEASURE == 3
      pthread_create(&thread[i], &attr, stack_measure_push_pop, (void*)&arg[i]);
#else
      pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#endif
    }
  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  clock_gettime(CLOCK_MONOTONIC, &stop);
  stack_check(stack);
  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
        printf("Thread %d time: %f\n", i, timediff(&t_start[i], &t_stop[i]));
    }
#endif

  return 0;
}
