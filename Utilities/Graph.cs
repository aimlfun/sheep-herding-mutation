using SheepHerderMutation.AI;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SheepHerderMutation.Utilities;

internal static class Graph
{
    internal static int s_scale = 10;

    /// <summary>
    /// Draws a rudimentary graph to a BitMap.
    /// </summary>
    /// <param name="performance"></param>
    /// <param name="neuralNetworkId"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <returns></returns>
    internal static Bitmap CreateSimpleImageOffPerformanceValues(int rank, List<int> performance, int neuralNetworkId, int width, int height)
    {
        int[] perfArray = performance.ToArray();

        s_scale = (int) (260f / LearnToHerd.s_wayPointsSheepNeedsToGoThru.Length);

        Bitmap bitmap = new(width, height);

        using Graphics graphics = Graphics.FromImage(bitmap);

        // blue background for mutated, grey for not.
        graphics.Clear(NeuralNetwork.s_networks[neuralNetworkId].Mutated ? Color.FromArgb(25, 25, 40) : Color.FromArgb(40, 40, 40));

        graphics.Flush();
        graphics.CompositingQuality = CompositingQuality.HighQuality;
        graphics.SmoothingMode = SmoothingMode.HighQuality;

        // add the title of the neural network id
        DrawTitleAtTopOfGraphContainingIdOfNeuralNetwork(neuralNetworkId, graphics, rank);

        using Pen penLines = new(Color.FromArgb(25, 200, 200, 200));
        penLines.DashStyle = DashStyle.Dot;

        using Font labelFont = new("Arial", 7);

        DrawGraphBackgroundOutline(graphics, penLines, labelFont);

        DrawRedLineIndicatingMax(perfArray, graphics);

        DrawOrangeLineIndicatingAvg(neuralNetworkId, graphics);

        DrawPerformanceDataLinePoints(perfArray, graphics);

        // I can never remember based on background colour, so added a "mutated" label to the graph.
        string text = NeuralNetwork.s_networks[neuralNetworkId].Mutated
            ? "- mutated -"
            : $"- survived {NeuralNetwork.s_networks[neuralNetworkId].GenerationOfLastMutation} ({NeuralNetwork.s_networks[neuralNetworkId].BestFitness()}/{Math.Round(NeuralNetwork.s_networks[neuralNetworkId].Score)}) -";
        WriteTextCentred(text, new Point(155, 5), graphics, labelFont, Brushes.Silver);

        double pct = NeuralNetwork.s_networks[neuralNetworkId].GenerationOfLastMutation == 0 ? 0 : Math.Round(100 * NeuralNetwork.s_networks[neuralNetworkId].BestFitness() / (NeuralNetwork.s_networks[neuralNetworkId].GenerationOfLastMutation + 1), 2);

        WriteTextCentred($"{pct}%", new Point(280, 8), graphics, labelFont, Brushes.Yellow);
        graphics.Flush();

        return bitmap;
    }

    /// <summary>
    /// Draws a horizontal line in orange, indicating the moving average.
    /// Note: It is based on a moving average (most recent 10% of the performance) not average of all data points.
    /// </summary>
    /// <param name="neuralNetworkId"></param>
    /// <param name="graphics"></param>
    private static void DrawOrangeLineIndicatingAvg(int neuralNetworkId, Graphics graphics)
    {
        // draw the line showing average
        float modeOrAverage = NeuralNetwork.s_networks[neuralNetworkId].AverageFitness() * s_scale;

        using Pen modeOrAverageLinePen = new(Color.FromArgb(150, 255, 165, 0));
        modeOrAverageLinePen.DashStyle = DashStyle.Dash;
        graphics.DrawLine(modeOrAverageLinePen, 31, 290 - modeOrAverage, 280, 290 - modeOrAverage);

        float sum = NeuralNetwork.s_networks[neuralNetworkId].Performance.Sum();
        float avg = NeuralNetwork.s_networks[neuralNetworkId].Performance.Count == 0 ? 0 : sum / NeuralNetwork.s_networks[neuralNetworkId].Performance.Count;

        using Pen averageLinePen = new(Color.FromArgb(150, 100, 100, 255));
        averageLinePen.DashStyle = DashStyle.Dash;

        graphics.DrawLine(averageLinePen, 31, 290 - avg * s_scale, 280, 290 - avg * s_scale);

    }

    /// <summary>
    /// Writes the ID of the network at the top of the graph.
    /// </summary>
    /// <param name="neuralNetworkId"></param>
    /// <param name="graphics"></param>
    private static void DrawTitleAtTopOfGraphContainingIdOfNeuralNetwork(int neuralNetworkId, Graphics graphics, int rank)
    {
        using Font titleFont = new("Arial", 9);

        WriteTextCentred($"Id: {neuralNetworkId}", new Point(155, 16), graphics, titleFont, Brushes.White);

        // show the rank of this neural network
        using Font rankFont = new("Arial", 12, FontStyle.Bold);
        graphics.DrawString($"#{rank}", rankFont, Brushes.White, 2, 8);
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="text"></param>
    /// <param name="position"></param>
    /// <param name="graphics"></param>
    /// <param name="font"></param>
    /// <param name="brush"></param>
    private static void WriteTextCentred(string text, Point position, Graphics graphics, Font font, Brush brush)
    {
        SizeF titleSize = graphics.MeasureString(text, font); // work out the size, so we can center it

        graphics.DrawString(text, font, brush, position.X - titleSize.Width / 2, position.Y);
    }

    /// <summary>
    /// Draw the background.
    ///   ...
    ///   4|---------
    ///   3|---------
    ///   2|---------
    ///    +----------
    /// </summary>
    /// <param name="graphics"></param>
    /// <param name="penLines"></param>
    /// <param name="f"></param>
    private static void DrawGraphBackgroundOutline(Graphics graphics, Pen penLines, Font f)
    {
        // one line per way point
        for (int y = s_scale; y < 250; y += s_scale)
        {
            graphics.DrawLine(penLines, 25, 290 - y, 280, 290 - y);

            string label = (y / s_scale).ToString();
            SizeF labelSize = graphics.MeasureString(label, f);
            graphics.DrawString(label, f, Brushes.White, 25 - labelSize.Width, 290 - y - labelSize.Height / 2);
        }

        using Pen edgePen = new(Color.FromArgb(60, 255, 255, 255));
        edgePen.DashStyle = DashStyle.Dot;

        graphics.DrawLine(edgePen, 30, 290, 280, 290); // horizontal base line
        graphics.DrawLine(edgePen, 30, 290, 30, 35); // left vertical line    
    }

    /// <summary>
    /// Plots the data-points on the graph.
    /// </summary>
    /// <param name="perfArray"></param>
    /// <param name="graphics"></param>
    private static void DrawPerformanceDataLinePoints(int[] perfArray, Graphics graphics)
    {
        List<Point> p = new();

        // add all the points so we can draw in 1 call to gdi
        int i = 0;

        for (int a = Math.Max(0, perfArray.Length - 80); a < perfArray.Length; a++)
        {
            p.Add(new Point(i * 3 + 31, 290 - Math.Min(260, s_scale * perfArray[a])));
            ++i;
        }

        using Pen pen = new(Color.White);
        pen.EndCap = LineCap.RoundAnchor;
        pen.StartCap = LineCap.RoundAnchor;

        // draw the graph of points
        if (p.Count > 1) graphics.DrawLines(pen, p.ToArray());
    }

    /// <summary>
    /// Draw a red horizontal line indicating the max this neural network reached.
    /// </summary>
    /// <param name="perfArray">The performance data.</param>
    /// <param name="graphics"></param>
    private static void DrawRedLineIndicatingMax(int[] perfArray, Graphics graphics)
    {
        // find the max across all of the data for this flock, so we can draw a "max" red line
        int max = -1;

        foreach (int z in perfArray) if (z * s_scale > max) max = Math.Min(260, s_scale * z);

        // draw line showing max
        if (max != -1)
        {
            using Pen maxLinePen = new(max > 250 ? Color.FromArgb(150, 0, 255, 0) : Color.FromArgb(150, 255, 0, 0));
            maxLinePen.DashStyle = DashStyle.Dash;
            graphics.DrawLine(maxLinePen, 31, 290 - max, 280, 290 - max);
        }
    }
}
