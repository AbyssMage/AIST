# MAE of ALl
for i in 0 5 6 7 71 21 22 23 24 27 31; do \
    python train.py --tr $i --tc 1;
done
for i in 66 68 70 7 71 42 43 48 22 23 24; do \
    python train.py --tr $i --tc 2;
done
for i in 66 68 70 7 71 42 43 22 24 28; do \
    python train.py --tr $i --tc 3;
done
for i in 70 7 71 42 22 24 25 26 28 29; do \
    python train.py --tr $i --tc 7;
done

# MSE of ALl
for i in 0 5 6 7 21 22 23 27 31; do \
    python train.py --tr $i --tc 1;
done
for i in 66 68 70 7 71 42 43 48 22 23 24; do \
    python train.py --tr $i --tc 2;
done
for i in 66 68 70 7 71 42 43 22 24 28; do \
    python train.py --tr $i --tc 3;
done
for i in 70 7 71 42 22 24 25 26 28 29; do \
    python train.py --tr $i --tc 7;
done

