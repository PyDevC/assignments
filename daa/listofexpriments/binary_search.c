# include<stdio.h>

int main(){
  int arr[] = {1,2,3,4,5,6,7,8,9,10};
  int high = 9;
  int low = 0;
  int mid;
  int search;
  scanf("%d",&search);

  while ( low <= high ){
    mid = low + (high - low)/2;
    if ( arr[mid] == search ){
      printf("element found at index %d", mid);
      break;
    } else if (arr[mid] < search){
      low += mid;
    } else{
      high -= mid;
    }
  }
  return 0;
}
