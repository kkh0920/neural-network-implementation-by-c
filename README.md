# Neural Network Implementation By C
- 필기체 숫자 인식을 위한 인공 신경망 알고리즘을 C언어를 통해 직접 구현

<img width="700" height="380" alt="image" src="https://github.com/user-attachments/assets/26b4b17e-efa0-4744-8841-ddd1e16830da" />

## Constant  
  ```c
  // 총 학습 데이터 4000개 (0 ~ 9, 각 숫자마다 400개 씩)
  #define TOTAL_TRAIN_DATA 4000
  #define TRAIN_DATA_PER_NUMBER 400 

  // 총 테스트 데이터 2000개 (0 ~ 9, 각 숫자마다 200개 씩)
  #define TOTAL_TEST_DATA 2000 
  #define TEST_DATA_PER_NUMBER 200

  // 입력층(100) -> 중간층(20) -> 출력층(10, one-hot vector: 숫자 예측값)
  #define INPUT_LAYER 100
  #define HIDDEN_LAYER 20
  #define OUTPUT_LAYER 10

  // 학습률 (0.5)
  #define LEARNING_RATE 0.5
  ```

## Variable
  ```c
  // Test data
  FILE *test_data = fopen("test.txt", "r");
  double test_input_vector[TOTAL_TEST_DATA][INPUT_LAYER + 1];
  double test_target_vector[TOTAL_TEST_DATA][OUTPUT_LAYER + 1];

  // Train data
  FILE *train_data = fopen("train.txt", "r");
  double train_input_vector[TOTAL_TRAIN_DATA][INPUT_LAYER + 1];
  double train_target_vector[TOTAL_TRAIN_DATA][OUTPUT_LAYER + 1];

  // Layer
  double X[INPUT_LAYER + 1]; // input
  double Z[HIDDEN_LAYER + 1]; // hidden
  double Y[OUTPUT_LAYER + 1]; // output
  
  // Weight
  double v[INPUT_LAYER + 1][HIDDEN_LAYER + 1]; // input -> hidden 가중치
  double w[HIDDEN_LAYER + 1][OUTPUT_LAYER + 1]; // hidden -> output 가중치
  ```

## Back Propagation
  ```c
  double delta[OUTPUT_LAYER + 1];

  for (int i = 1; i <= OUTPUT_LAYER; i++) {
      delta[i] = (target_vector[i] - Y[i]) * (Y[i] * (1 - Y[i]));
      for (int j = 0; j <= HIDDEN_LAYER; j++) {
          w[j][i] += LEARNING_RATE * delta[i] * Z[j];
      }
  }

  for (int i = 1; i <= HIDDEN_LAYER; i++) {
      double delta_hidden = 0.0;
      for (int j = 1; j <= OUTPUT_LAYER; j++) {
          delta_hidden += delta[j] * w[i][j];
      }
      delta_hidden *= Z[i] * (1 - Z[i]);

      for (int j = 0; j <= INPUT_LAYER; j++) {
          v[j][i] += LEARNING_RATE * delta_hidden * X[j];
      }
  }
  ```

### 1. 수식 정의
$$ 
J = \frac{1}{2} (Y - t)^2\ 
$$

$$
Y = \sigma(Z w)
$$

$$
Z = \sigma(X v)
$$

### 2. 비용 함수 J를 가중치 v, w에 대해 미분
$$
\frac{\partial J}{\partial w} = \frac{\partial J}{\partial Y} \cdot \frac{\partial Y}{\partial w} = (Y - t) \cdot Y (1 - Y) \cdot Z
$$

$$
\frac{\partial J}{\partial v} = \frac{\partial J}{\partial Y} \cdot \frac{\partial Y}{\partial Z} \cdot \frac{\partial Z}{\partial v} = (Y - t) \cdot Y (1 - Y) \cdot w \cdot Z (1 - Z) \cdot X
$$

### 3. 경사 하강법을 통해 가중치 조정
$$
w \gets w - \eta \frac{\partial J}{\partial w}
$$

$$
v \gets v - \eta \frac{\partial J}{\partial v}
$$

## Result
<img width="450" height="300" alt="image" src="https://github.com/user-attachments/assets/84b3cf40-b0a7-4283-8843-82dee93bf598" />
