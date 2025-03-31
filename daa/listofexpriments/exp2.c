/*merge sort*/
#include <stdio.h>
#include <stdlib.h>

void merge(int *arr, int left, int mid, int right) {
  int num1 = mid - left + 1;
  int num2 = right - mid;
  int Left[num1], Right[num2];

  int i, j;
  for (i = 0; i < num1; i++) {
    Left[i] = arr[left + 1];
  }
  for (j = 0; j < num2; j++) {
    Right[j] = arr[mid + 1 + j];
  }

  i = 0;
  j = 0;
  int k = left;
  while (i < num1 && j < num2) {
    if (Left[i] <= Right[j]) {
      arr[k] = Left[i];
      i++;
    } else {
      arr[k] = Right[j];
      j++;
    }
    k++;
  }

  while (i < num1) {
    arr[k] = Left[i];
    i++;
    k++;
  }

  while (j < num2) {
    arr[k] = Right[j];
    j++;
    k++;
  }
}

void mergesort(int *arr, int left, int right) {
  if (left < right) {
    int mid = left + (right - left) / 2;
    mergesort(arr, left, mid);
    mergesort(arr, mid + 1, right);
    merge(arr, left, mid, right);
  }
}

int main(void) {

  printf("Anushk Sharma\n");
  int arr[5] = {11, 2, 32, 4, 5};
  int *ptr = malloc(sizeof(int) * 5);
  int size = sizeof(arr) / sizeof(arr[0]);
  ptr = &arr[0];
  int i;

  for (i = 0; i < 5; i++) {
    printf("%d ", *ptr++);
  }
  ptr = &arr[0];

  mergesort(ptr, 0, size-1);

  // sorted array
  printf("\nsorted array\n");

  for (i = 0; i < 5; i++) {
    printf("%d ", *ptr++);
  }

  return 0;
}
