import { Component } from '@angular/core';
import {AppStateService} from "../../../../StateManagement/app-state.service";
import {map} from "rxjs";

@Component({
  selector: 'app-score-presentation',
  templateUrl: './score-presentation.component.html',
  styleUrls: ['./score-presentation.component.css']
})
export class ScorePresentationComponent {


  objectOneScore = 0.5;

  constructor(public state: AppStateService) {
    this.state.state$.pipe(
      map(state => state.objectOneScore)
    ).subscribe(score => this.objectOneScore = score)
  }
}
