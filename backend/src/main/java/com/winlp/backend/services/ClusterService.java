package com.winlp.backend.services;

import com.winlp.backend.dtos.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * This service is responsible for communication with the rest of the APIs in the cluster.
 */
@Service
public class ClusterService {

    @Autowired
    RestTemplate restTemplate;

    @Value("${CQI_URL}")
    private String cqiUrl;

    @Value("${OAI_URL}")
    private String oaiUrl;

    @Value("${CAM_URL}")
    private String camUrl;

    @Value("${CSI_URL}")
    private String csiUrl;

    @Value("${CQAS_URL}")
    private String cqasUrl;


    /**
     * This method is responsible for calling the comparative question detection API.
     *
     * @param question The question to be checked.
     * @return True if the question is comparative, false otherwise.
     */
    public boolean isComparative(String question) {
        System.out.println("===========================================");
        System.out.println("making CQI request to: " + cqiUrl + "/" + question);
        boolean ans = Boolean.TRUE.equals(restTemplate.getForObject(cqiUrl + "/" + question, boolean.class));
        System.out.println("CQI response: " + ans);
        System.out.println("===========================================");
        return ans;
    }

    /**
     * This method is responsible for calling the object and aspect detection API.
     *
     * @param question The question to be checked.
     * @return A list of lists of strings, where the first list contains the objects and the second list contains the aspects.
     */
    public ObjectsAndAspectsResponse getObjectsAndAspects(String question, boolean faster) {
        // Call the object and aspect detection API
        System.out.println("===========================================");
        System.out.println("making OAI request to: " + oaiUrl + "/" + faster + "/" + question);
        ObjectsAndAspectsResponse response = restTemplate.getForObject(oaiUrl + "/" + faster + "/" + question, ObjectsAndAspectsResponse.class);
        System.out.println("OAI response: " + response);
        System.out.println("===========================================");
        return response;
    }

    /**
     * This method is responsible for calling the comparative argumentation machine API.
     *
     * @param objects The objects to be compared.
     * @param aspects The aspects to be compared.
     * @return A list of strings, where each string is an argument about the comparison.
     */
    public CamResponse getComparisonArguments(List<String> objects, List<String> aspects, boolean faster) {
        StringBuilder urlBuilder = new StringBuilder(camUrl + "?fs=" + faster + "&objectA=" + objects.get(0) + "&objectB=" + objects.get(1));
        for (int i = 0; i < aspects.size(); i++) {
            urlBuilder.append("&aspect").append(i + 1).append("=").append(aspects.get(i)).append("&weight").append(i + 1).append("=").append(1000000);
        }

        String url = urlBuilder.toString();
        System.out.println("===========================================");
        System.out.println("making CAM request to: " + url);
        System.out.println("===========================================");
        CamApiResponse response = restTemplate.getForObject(urlBuilder.toString(), CamApiResponse.class);

        double score = response.object1.totalPoints / (response.object1.totalPoints + response.object2.totalPoints);

        var args1 = response.object1.sentences;
        var args2 = response.object2.sentences;

        Collections.sort(args1);
        Collections.sort(args2);

        List<Argument> firstObjectArguments = args1.stream().map(sentence -> new Argument(sentence.text, sentence.id_pair.keySet().stream().toList().get(0))).toList();
        List<Argument> secondObjectArguments = args2.stream().map(sentence -> new Argument(sentence.text, sentence.id_pair.keySet().stream().toList().get(0))).toList();

        return new CamResponse(score, firstObjectArguments, secondObjectArguments);
    }

    /**
     * This method is responsible for calling the sentence identification API.
     * @param request The request containing the objects, aspects and arguments.
     * @return A response containing the sentences.
     */
    public SentenceIdentificationResponse classifySentences(SentenceIdentificationRequest request) {
        System.out.println("===========================================");
        System.out.println("making CSI request to: " + csiUrl);
        System.out.println("request: " + request);
        SentenceIdentificationResponse response = restTemplate.postForObject(csiUrl, request, SentenceIdentificationResponse.class);
        System.out.println("CSI response: " + response);
        System.out.println("===========================================");
        return response;
    }


    /**
     * This method is responsible for calling the arguments summarization API.
     *
     * @param arguments The arguments to be summarized.
     * @return A string containing the summary of the arguments.
     */
    public String getSummary(List<String> arguments, String object1, String object2) {
        System.out.println("===========================================");
        System.out.println("making CQAS request to: " + cqasUrl);
        System.out.println("arguments: " + arguments);
        String Summary = restTemplate.postForObject(cqasUrl, new SummaryRequest(object1, object2, arguments), String.class);
        System.out.println("CQAS response: " + Summary);
        System.out.println("===========================================");
        return Summary;
    }
}