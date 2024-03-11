package com.winlp.backend.dtos;

import java.util.List;
import java.util.Map;

public class CamApiResponse {
    public List<String> extractedAspectsObject1;
    public List<String> extractedAspectsObject2;
    public CamObject object1;
    public CamObject object2;
    public int sentenceCount;
    public String winner;
    public static class CamObject {
        public String name;

        public Map<String, Double> points;
        public List<CamSentence> sentences;
        public double totalPoints;

        public static class CamSentence implements Comparable<CamSentence> {
            public double CAM_score;
            public double ES_score;
            public double confidence;
            public List<String> context_aspects;
            public Map<String, Integer> id_pair;
            public String text;


            @Override
            public int compareTo(CamSentence o) {
                if (this.CAM_score > o.CAM_score)
                    return -1;
                else if (this.CAM_score < o.CAM_score)
                    return 1;
                return 0;
            }
        }
    }

}
