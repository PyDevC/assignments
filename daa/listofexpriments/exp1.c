/*write a program for insertion sort*/

#include <stdio.h>
#include <stdlib.h>

void insertion(int *arr, int size) {
  int i, j, key;

  for (i = 1; i < size; i++) {
    key = arr[i];
    j = i - 1;
    while (j >= 0 && (key < arr[j])) {
      arr[j + 1] = arr[j];
      j -= 1;
    }
    arr[j + 1] = key;
  }
}

int main(void) {

  int arr[5] = {11, 2, 32, 4, 5};
  int *ptr = malloc(sizeof(int) * 5);
  ptr = &arr[0];
  int i;

  for (i = 0; i < 5; i++) {
    printf("%d ", *ptr++);
  }
  ptr = &arr[0];

  insertion(ptr, 5);

  // sorted array
  printf("\nsorted array\n");

  for (i = 0; i < 5; i++) {
    printf("%d ", *ptr++);
  }

  return 0;
}
