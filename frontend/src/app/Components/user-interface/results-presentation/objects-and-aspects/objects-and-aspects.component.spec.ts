import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ObjectsAndAspectsComponent } from './objects-and-aspects.component';

describe('ObjectsAndAspectsComponent', () => {
  let component: ObjectsAndAspectsComponent;
  let fixture: ComponentFixture<ObjectsAndAspectsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ObjectsAndAspectsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ObjectsAndAspectsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
