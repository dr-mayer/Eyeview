/**
 * Created by Nathan on 4/21/16.
 */


public class TestFits {

    public static void main(String[] arg) {

        double[] bids = new double[]{8120, 22773, 398715, 35735, 10755,
                198005, 6164, 105051, 75657, 3513, 97248, 61627, 475294,
                41688, 113351, 520531, 5473, 164165, 212204, 48636, 11967,
                83633, 8695, 70, 14332, 3834, 73912};

        double[] wins = new double[]{1122, 4744, 15696, 4715, 1366, 875, 1202,
                5702, 6827, 201, 12984, 11514, 4325, 8368, 12864, 26104, 868,
                23806, 12307, 9521, 1044, 12546, 1733, 5, 1902, 274, 9443};

        double[] bid_price = new double[]{12.4879, 11.4586, 4.43805, 40.3067,
                13.5453, 0.907924, 51.472, 5.40474, 6.49524, 90.0051, 27.2161,
                8.43601, 2.57214, 10.4368, 18.0747, 3.45986, 47.3593, 21.7797,
                1.37244, 9.47295, 81.0761, 7.52419, 36.7029, 68.623, 14.4531,
                72.4406, 30.2934};

        double[] win_price = new double[]{7.43716, 7.33681, 3.49868, 17.3109, 8.4819,
                0.985474, 11.575, 4.32108, 4.75859, 16.3166, 11.1103, 5.96542, 2.69332,
                6.71311, 9.45762, 2.91417, 13.1598, 10.4716, 1.05465, 6.32676, 13.0336,
                5.48389, 11.9986, 6.698, 9.05815, 12.7256, 10.7394};

        double[] win_price_sqr = new double[]{63.8885, 60.5911, 12.8723, 365.623, 83.1226,
                0.97562, 201.281, 19.8493, 24.5955, 698.034, 154.738, 38.6095, 7.50718,
                50.427, 107.001, 8.64475, 228.273, 132.437, 1.13475, 43.6577, 309.952, 32.2896,
                191.671, 70.5044, 93.9369, 277.295, 151.294};

        double[] totalWp = win_price;
        double[] totalBp = bid_price;
        double[] totalWpSqr = win_price_sqr;

        for(int i=0; i<bids.length;i++) {
            totalWp[i]    = totalWp[i]*wins[i];
            totalBp[i]    = totalBp[i]*bids[i];
            totalWpSqr[i] = totalWpSqr[i]*wins[i];
        }

        WinPrice wp = new WinPrice(wins, bids, totalBp, totalWp, totalWpSqr, 0);
        WinRate  wr = new WinRate(bids, wins, totalBp, 0);

        double wr_ll   = wr.ReducedLogLikelihood();
        double wp_chi2 = wp.ReducedChi2();

        double[] wr_bf = wr.BestFit();
        double[] wp_bf = wp.BestFit();

        System.out.println("Win Rate fit parameters (p1=" + wr_bf[0] + ", p2="
                + wr_bf[1] + ", p3=" + wr_bf[2] + "), with a Reduced LogLikelihood of: " + wr_ll);
        System.out.println("Win Price fit parameters (" + wp_bf[0] + ", "
                + wp_bf[1] + ", " + wp_bf[2] + "), with a Reduced Chi Squared of: " + wp_chi2);


    }
}
