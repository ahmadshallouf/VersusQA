package com.winlp.backend.dtos;


import java.util.List;

public record ObjectsAndAspectsResponse(List<String> objects,
                                        List<String> aspects) {
}
