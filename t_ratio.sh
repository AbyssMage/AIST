for i in 21 22 23 24 27; do \
    python train_test_ratio_aist.py --tr $i;
done
for i in 66 68 70 7 71 42 43 22 24 28; do \
    python train.py --tr $i --tc 3;
done
for i in 66 68 70 7 71 42 43 48 22 23 24; do \
    python train.py --tr $i --tc 2;
done
for i in 0 5 6 7 71 21 22 23 24 27 31; do \
    python train.py --tr $i --tc 1;
done
# For MSE
for i in 65 66 68 70 7 42 43 24 28; do \
    python train.py --tr $i --tc 3;
done
for i in 70 42 22 24 25 26 28 29; do \
    python train.py --tr $i --tc 7;
done
for i in 5 6 7 71 21 22 23 24 27 31; do \
    python train.py --tr $i --tc 1;
done
for i in 66 68 70 7 71 42 43 48 22 23 24; do \
    python train.py --tr $i --tc 2;
done

