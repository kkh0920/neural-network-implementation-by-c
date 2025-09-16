#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
// #include <windows.h> // For Windows system("cls")

#define TOTAL_TRAIN_DATA 4000
#define TRAIN_DATA_PER_NUMBER 400

#define TOTAL_TEST_DATA 2000
#define TEST_DATA_PER_NUMBER 200

#define INPUT_LAYER 100
#define HIDDEN_LAYER 20
#define OUTPUT_LAYER 10

#define LEARNING_RATE 0.5

#define MAX_EPOCH 100

void read_data(FILE *file, 
               double input_vector[][INPUT_LAYER + 1], 
               double target_vector[][OUTPUT_LAYER + 1], 
               int total_data);

void initialize_weights(double v[][HIDDEN_LAYER + 1], double w[][OUTPUT_LAYER + 1]);

void forward_propagation(double *input, 
                         double *X, double *Z, double *Y, 
                         double v[][HIDDEN_LAYER + 1], double w[][OUTPUT_LAYER + 1]);

void backward_propagation(double *target_vector, 
                          double *X, double *Z, double *Y,
                          double v[][HIDDEN_LAYER + 1], double w[][OUTPUT_LAYER + 1]);

double precision_ratio(double input_vector[][INPUT_LAYER + 1], 
                       double target_vector[][OUTPUT_LAYER + 1], 
                       double *X, double *Z, double *Y, 
                       double v[][HIDDEN_LAYER + 1], double w[][OUTPUT_LAYER + 1], 
                       int total_data);

double sigmoid(double x);

int main() {
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

    // Precision
    double goal_precision = -1;
    double test_precision = 0;
    double train_precision = 0;

    // Epoch
    int epoch = 1;

    srand(time(NULL));

    /*  1. 데이터 불러오기  */
    read_data(train_data, train_input_vector, train_target_vector, TOTAL_TRAIN_DATA);
    read_data(test_data, test_input_vector, test_target_vector, TOTAL_TEST_DATA);

    /*  2. (-1) ~ (+1) 사이의 랜덤한 값으로 가중치 초기화  */
    initialize_weights(v, w); 
    
    /*  3. 목표 정확도 입력  */
    while (goal_precision < 0 || goal_precision > 100) {
        printf("\nEnter the goal precision (0 ~ 100): ");
        scanf("%lf", &goal_precision);
    }

    printf("\nStart training...\n\n");
    
    /*  4. 목표 정확도에 달성할 때까지 숫자 데이터 학습  */
    while (test_precision < goal_precision && epoch <= MAX_EPOCH) {
        int startIndex = 0;
        int count = TRAIN_DATA_PER_NUMBER;

        /* 
           4.1. 현재 데이터는 0 ~ 9까지의 숫자가 400개씩 순서대로 정렬되어 있다. 
                따라서, 0 ~ 9의 숫자를 균등하게 학습시키기 위해, 인덱스를 400개씩 건너뛰면서 학습시킨다. 
        */
        while (count--) {
            for (int dataIndex = startIndex++; dataIndex < TOTAL_TRAIN_DATA; dataIndex += TRAIN_DATA_PER_NUMBER) {
                forward_propagation(train_input_vector[dataIndex], X, Z, Y, v, w);
                backward_propagation(train_target_vector[dataIndex], X, Z, Y, v, w);
            }   
        }

        /*  4.2. 매 epoch 마다 정확도를 계산하고 출력  */
        train_precision = precision_ratio(train_input_vector, train_target_vector, X, Z, Y, v, w, TOTAL_TRAIN_DATA);
        test_precision = precision_ratio(test_input_vector, test_target_vector, X, Z, Y, v, w, TOTAL_TEST_DATA);
        
        system("clear"); // Linux / Mac
        // system("cls"); // Windows
        printf("\n #############  Training  ############# \n\n");
        printf("    Epoch: %d (maximum: %d)\n\n", epoch++, MAX_EPOCH);
        printf("    Train precision: %.2f%%\n\n", train_precision);
        printf("    Test precision: %.2f%% (goal: %.2f%%)\n\n", test_precision, goal_precision);
        printf(" ###################################### \n\n");
    }
    
    printf("\n Training completed!\n\n");

    fclose(train_data);
    fclose(test_data);

    return 0;
}

void read_data(FILE *file, double input_vector[][INPUT_LAYER + 1], double target_vector[][OUTPUT_LAYER + 1], int total_data) {
    char line[1024];
    for (int i = 0; i < total_data; i++) {
        // 첫 번째 줄은 정답 레이블
        fgets(line, sizeof(line), file);
        for (int j = 1; j <= OUTPUT_LAYER; j++) {
            target_vector[i][j] = 0.0;
        }
        target_vector[i][atoi(line) + 1] = 1.0;
        // 두 번째 줄부터는 열 개의 줄에 걸쳐서 0 ~ 1 사이의 값이 들어온다.
        input_vector[i][0] = 1.0; // bias
        int index = 1;
        for (int j = 0; j < 10; j++) {
            fgets(line, sizeof(line), file);
            char *token = strtok(line, " ");
            while (strcmp(token, "\r\n")) {
                input_vector[i][index++] = atof(token);
                token = strtok(NULL, " ");
            }
        }
    }
}

void initialize_weights(double v[][HIDDEN_LAYER + 1], double w[][OUTPUT_LAYER + 1]) {
    for (int i = 0; i < INPUT_LAYER + 1; i++) {
        for (int j = 0; j < HIDDEN_LAYER + 1; j++) {
            v[i][j] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    for (int i = 0; i < HIDDEN_LAYER + 1; i++) {
        for (int j = 0; j < OUTPUT_LAYER + 1; j++) {
            w[i][j] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
}

void forward_propagation(double *input, double *X, double *Z, double *Y, 
                         double v[][HIDDEN_LAYER + 1], double w[][OUTPUT_LAYER + 1]) {
    // fill input layer
    for (int i = 1; i <= INPUT_LAYER; i++) {
        X[i] = input[i];
    }
    // input layer -> hidden layer
    for (int i = 1; i <= HIDDEN_LAYER; i++) {
        Z[i] = v[0][i];
        for (int j = 1; j <= INPUT_LAYER; j++) {
            Z[i] += X[j] * v[j][i];
        }
        Z[i] = sigmoid(Z[i]);
    }
    // hidden layer -> output layer
    for (int i = 1; i <= OUTPUT_LAYER; i++) {
        Y[i] = w[0][i];
        for (int j = 1; j <= HIDDEN_LAYER; j++) {
            Y[i] += Z[j] * w[j][i];
        }
        Y[i] = sigmoid(Y[i]);
    }
}

void backward_propagation(double *target_vector, double *X, double *Z, double *Y, 
                          double v[][HIDDEN_LAYER + 1], double w[][OUTPUT_LAYER + 1]) {
    double delta[OUTPUT_LAYER + 1];
    for (int i = 1; i <= OUTPUT_LAYER; i++) {
        // Y[i]는 sigmoid 함수의 결과이므로, 바로 미분식을 적용할 수 있다.
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
        // Z[i]는 sigmoid 함수의 결과이므로, 바로 미분식을 적용할 수 있다.
        delta_hidden *= Z[i] * (1 - Z[i]);

        for (int j = 0; j <= INPUT_LAYER; j++) {
            v[j][i] += LEARNING_RATE * delta_hidden * X[j];
        }
    }
}

double precision_ratio(double input_vector[][INPUT_LAYER + 1], double target_vector[][OUTPUT_LAYER + 1], 
                     double *X, double *Z, double *Y, 
                     double v[][HIDDEN_LAYER + 1], double w[][OUTPUT_LAYER + 1], int total_data) {
    int correct = 0;
    for (int i = 0; i < total_data; i++) {
        forward_propagation(input_vector[i], X, Z, Y, v, w);
        int maxIndex = 1;
        for (int i = 2; i <= OUTPUT_LAYER; i++) {
            if (Y[i] > Y[maxIndex]) {
                maxIndex = i;
            }
        }
        if (target_vector[i][maxIndex] == 1.0) {
            correct++;
        }
    }
    return ((double) correct / total_data) * 100;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}